# === PHYSPT: imports ===
from hmr2.physpt.adapter import PhysPTWrapper
from hmr2.physpt.utils import WindowBuffer, pack_q

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import inspect
import functools

import os
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

warnings.filterwarnings('ignore')
log = get_pylogger(__name__)


# =========================
#  HMR2.0 / 纹理采样封装
# =========================
class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256  # self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.  # self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                           image_size=self.img_size,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=False)

    def forward(self, x):
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh
        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]  # N
            bmap_flat = bmap[valid_mask, :]  # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :]  # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat)  # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor)  # B,N,3

        # Project map_verts to image using K,R,t
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3]  # B,N,2
        map_verts_depth = map_verts[:, :, 2]  # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                          face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                          # textures=texture_atlas_rgb,
                                          mode='depth',
                                          K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:, None, :, :],
                                                             map_verts_proj[:, None, :, :])  # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1)  # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:, None, :, :]], dim=1)  # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:, None, :, :])  # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2)  # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4)  # B,N
        img_rgba_at_proj[:, 3, :][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image': uv_image,
            'uv_vector': self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


# ==========================================
#  继承 PHALP：接 PhysPT 的两处钩子（方案 A）
# ==========================================
class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

        # ---------- PhysPT 全局开关 & 参数 ----------
        self.PHYS_ENABLED = True  # 临时关掉切 False
        self.PHYS_T = 16  # 窗口长度
        self.PHYS_N_FUTURE = 1  # 外推帧数（1 即可）
        # 代价项权重：原有 cost（IoU/外观等）在 PHALP 内部已算好；我们叠加 Eq/Ec
        self.W_EQ = 1.0  # 参数空间 L2（运动先验）
        self.W_EC = 0.0  # 接触一致性（先置 0，Step 10 再开）

        # PhysPT 适配器与滑窗缓存
        try:
            self.phys_wrapper = PhysPTWrapper("third_party/PhysPT", device="cuda")
            log.info("[PhysPT] 已初始化第三方模型（third_party/PhysPT）")
        except Exception as e:
            self.PHYS_ENABLED = False
            log.warning(f"[PhysPT] 初始化失败，已自动关闭 PhysPT：{e}")
        self.phys_buffers: Dict[int, WindowBuffer] = {}  # track_id -> WindowBuffer

        # 尝试挂钩：把 PhysPT 先验加入“代价矩阵构建”，并在“匹配回写后”做窗口精炼
        self._install_physpt_hooks()

    # 使用 HMR2 替换老 HMR
    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    # 保持你原有的 bbox padding 行为
    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        ) = super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )

    # ------------------ PhysPT 钩子安装 ------------------
    def _install_physpt_hooks(self):
        """
        动态寻找 PHALP 中“构建代价矩阵”和“匹配后更新”的方法名并包裹。
        不同版本命名可能不同，这里罗列常见候选并逐一尝试；找不到则降级为不启用 PhysPT。
        """
        if not self.PHYS_ENABLED:
            return

        # 候选方法名（不同版本可能不同）
        cost_fn_candidates = [
            "build_cost_matrix", "compute_cost", "get_cost",
            "compute_association_cost", "association_cost",
        ]
        post_update_candidates = [
            "update_matched", "post_update", "after_update",
            "update_tracklets", "update_tracks",
        ]

        # 挂钩：代价矩阵构建
        for name in cost_fn_candidates:
            if hasattr(self, name) and callable(getattr(self, name)):
                self._orig_cost_fn_name = name
                orig = getattr(self, name)
                wrapped = self._wrap_cost_with_physpt(orig)
                setattr(self, name, wrapped)
                log.info(f"[PhysPT] 已挂钩关联代价函数: {name}(...)-> cost += W_EQ*Eq + W_EC*Ec")
                break
        else:
            log.warning("[PhysPT] 未找到可挂钩的代价矩阵函数（已跳过关联前先验）。"
                        "可在 HMR2_4dhuman._install_physpt_hooks 里把方法名改成你本地版本。")
            self._orig_cost_fn_name = None

        # 挂钩：匹配后更新
        for name in post_update_candidates:
            if hasattr(self, name) and callable(getattr(self, name)):
                self._orig_post_update_name = name
                orig = getattr(self, name)
                wrapped = self._wrap_post_update_with_physpt(orig)
                setattr(self, name, wrapped)
                log.info(f"[PhysPT] 已挂钩匹配后更新函数: {name}(matches, ...)-> refine_window 回写")
                break
        else:
            log.warning("[PhysPT] 未找到可挂钩的匹配后更新函数（已跳过窗口精炼回写）。"
                        "可在 HMR2_4dhuman._install_physpt_hooks 里把方法名改成你本地版本。")
            self._orig_post_update_name = None

    # ------------------ 工具：抽取一帧 SMPL -> 75维 ------------------
    @staticmethod
    def _extract_q_beta_from_track(trk) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        试着从不同版本的 track/state 里拿到 transl(3), global_orient(3 axis-angle), body_pose(69) 与 betas(10)。
        返回 (q_t: (75,), beta:(10,))；失败则返回 None。
        """
        try:
            st = getattr(trk, "state", trk)
            T = np.asarray(st.transl).reshape(-1)  # (3,)
            R = np.asarray(st.global_orient).reshape(-1)  # (3,)
            th = np.asarray(st.body_pose).reshape(-1)  # (69,)
            beta = np.asarray(st.betas).reshape(-1)  # (10,)
            if T.size == 3 and R.size == 3 and th.size == 69 and beta.size >= 10:
                return pack_q(T, R, th), beta[:10]
        except Exception:
            pass
        # 备选：有些版本把 smpl 放在 trk.smpl 里
        try:
            smpl = trk.smpl
            T = np.zeros(3, dtype=np.float32)  # 若无平移，置零（PHALP 常用相机平移）
            R = np.asarray(smpl.get("global_orient")).reshape(-1)
            th = np.asarray(smpl.get("body_pose")).reshape(-1)
            beta = np.asarray(smpl.get("betas")).reshape(-1)
            if R.size == 3 and th.size == 69 and beta.size >= 10:
                return pack_q(T, R, th), beta[:10]
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_q_from_det(det) -> Optional[np.ndarray]:
        """
        从 detection 取出 HMR2/PHALP 形成的 SMPL 参数并封装 75维 q。
        可能来源：det.smpl, det.pose_smpl, det.pred_smpl_params 等。
        """
        # 常见：PHALP 将 smpl 塞在 det.smpl 字典
        try:
            smpl = det.smpl
            T = np.zeros(3, dtype=np.float32)  # 单帧观测常无世界平移；设零不影响关联相对比较
            R = np.asarray(smpl.get("global_orient")).reshape(-1)
            th = np.asarray(smpl.get("body_pose")).reshape(-1)
            if R.size == 3 and th.size == 69:
                return pack_q(T, R, th)
        except Exception:
            pass
        # HMR2 输出可能是 det.pose_smpl（张量或字典）
        for key in ["pose_smpl", "pred_smpl_params"]:
            try:
                ps = getattr(det, key)
                # 期望字段名
                R = np.asarray(ps["global_orient"]).reshape(-1)
                th = np.asarray(ps["body_pose"]).reshape(-1)
                T = np.zeros(3, dtype=np.float32)
                if R.size == 3 and th.size == 69:
                    return pack_q(T, R, th)
            except Exception:
                continue
        return None

    # ------------------ 包裹：在构建代价矩阵时加 Eq/Ec ------------------
    def _wrap_cost_with_physpt(self, orig_fn):
        """
        orig_fn: 原始的构建代价矩阵函数，签名可能是：
            cost = orig_fn(live_tracks, detections, *args, **kwargs)
            或 orig_fn(tracks=..., detections=..., ...)
        我们将：cost += W_EQ * Eq + W_EC * Ec
        """

        @functools.wraps(orig_fn)
        def wrapper(*args, **kwargs):
            cost = orig_fn(*args, **kwargs)
            if not self.PHYS_ENABLED:
                return cost

            # —— 从调用参数里取 live_tracks / detections —— #
            live_tracks = None
            detections = None

            # 位置参数 & 关键字参数同时尝试
            sig = inspect.signature(orig_fn)
            bound = None
            try:
                # 用原始 fn 的签名绑定（注意此处必须用 orig_fn 的签名，而不是 wrapper）
                bound = sig.bind_partial(self, *args, **kwargs) if 'self' in sig.parameters else sig.bind_partial(*args,
                                                                                                                  **kwargs)
            except Exception:
                pass

            if bound:
                for k, v in bound.arguments.items():
                    if isinstance(v, (list, tuple)):
                        # 粗略猜测
                        if live_tracks is None and len(v) > 0 and hasattr(v[0], "id"):
                            live_tracks = v
                        elif detections is None and len(v) >= 0:
                            detections = v
                    elif k in ("tracks", "live_tracks", "tracklets"):
                        live_tracks = v
                    elif k in ("detections", "dets"):
                        detections = v

            # 兜底：从 args/kwargs 尝试
            if live_tracks is None:
                for a in args:
                    if isinstance(a, (list, tuple)) and len(a) > 0 and hasattr(a[0], "id"):
                        live_tracks = a;
                        break
            if detections is None:
                for a in args:
                    if isinstance(a, (list, tuple)):
                        detections = a;
                        break
            live_tracks = live_tracks or kwargs.get("tracks") or kwargs.get("live_tracks") or kwargs.get("tracklets")
            detections = detections or kwargs.get("detections") or kwargs.get("dets")

            if live_tracks is None or detections is None:
                log.warning("[PhysPT] 未能解析 cost 函数入参（tracks/detections），已跳过 PhysPT 先验。")
                return cost

            # —— 准备每条 track 的“下一帧预测” q_pred —— #
            pred_next_by_tid: Dict[int, np.ndarray] = {}
            det_qhats: List[Optional[np.ndarray]] = []

            # 先把 det 的 q 打包（避免双重循环里反复提取）
            for det in detections:
                det_qhats.append(self._extract_q_from_det(det))

            for trk in live_tracks:
                # 滑窗推进
                qb = self._extract_q_beta_from_track(trk)
                if qb is None:
                    continue
                q_t, beta = qb
                buf = self.phys_buffers.get(trk.id)
                if buf is None:
                    buf = WindowBuffer(T=self.PHYS_T)
                    self.phys_buffers[trk.id] = buf
                buf.push(q_t, beta, feet_points_world=None)

                if buf.ready():
                    q_win, b, plane = buf.as_arrays()
                    try:
                        q_future = self.phys_wrapper.predict_next(q_win, b, plane, n_future=self.PHYS_N_FUTURE)
                        pred_next_by_tid[trk.id] = q_future[-1]  # (75,)
                    except Exception as e:
                        log.warning(f"[PhysPT] 预测下一帧失败（已跳过该轨迹）: {e}")

            # —— 把 Eq/Ec 叠加到 cost 上 —— #
            try:
                import numpy as _np
                cost_np = _np.asarray(cost, dtype=_np.float32)
                for i, trk in enumerate(live_tracks):
                    q_pred = pred_next_by_tid.get(trk.id, None)
                    if q_pred is None:
                        continue
                    for j, _det in enumerate(detections):
                        dq = det_qhats[j]
                        if dq is None:
                            continue
                        Eq = float(_np.linalg.norm(q_pred - dq))
                        Ec = 0.0  # Step 10 再加入接触一致性
                        cost_np[i, j] += self.W_EQ * Eq + self.W_EC * Ec
                return cost_np
            except Exception as e:
                log.warning(f"[PhysPT] 叠加成本失败（已使用原始 cost）: {e}")
                return cost

        return wrapper

    # ------------------ 包裹：匹配后进行窗口精炼并回写 ------------------
    def _wrap_post_update_with_physpt(self, orig_fn):
        """
        orig_fn: 匹配后更新（签名示例）：
            orig_fn(matches, live_tracks, detections, *args, **kwargs)
        我们在调用原函数后，对匹配成功的轨迹做：
            - 把本帧观测 push 进滑窗
            - 若 ready()，调用 refine_window，回写 transl/global_orient/body_pose
        """

        @functools.wraps(orig_fn)
        def wrapper(*args, **kwargs):
            out = orig_fn(*args, **kwargs)

            if not self.PHYS_ENABLED:
                return out

            # 解析参数
            matches = None
            live_tracks = None
            detections = None

            sig = inspect.signature(orig_fn)
            bound = None
            try:
                bound = sig.bind_partial(self, *args, **kwargs) if 'self' in sig.parameters else sig.bind_partial(*args,
                                                                                                                  **kwargs)
            except Exception:
                pass

            if bound:
                for k, v in bound.arguments.items():
                    if k in ("matches", "assignments"):
                        matches = v
                    elif k in ("tracks", "live_tracks", "tracklets"):
                        live_tracks = v
                    elif k in ("detections", "dets"):
                        detections = v

            # 兜底
            matches = matches or kwargs.get("matches") or kwargs.get("assignments")
            live_tracks = live_tracks or kwargs.get("tracks") or kwargs.get("live_tracks") or kwargs.get("tracklets")
            detections = detections or kwargs.get("detections") or kwargs.get("dets")

            if matches is None or live_tracks is None:
                return out

            # 对每个成功匹配的轨迹，完成 push + refine + 回写
            for pair in matches:
                try:
                    # 有的版本 pair 是 (ti, dj)，也可能是对象或字典
                    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                        ti, dj = int(pair[0]), int(pair[1])
                        trk = live_tracks[ti]
                    elif hasattr(pair, "track_idx") and hasattr(pair, "det_idx"):
                        trk = live_tracks[pair.track_idx]
                        dj = pair.det_idx
                    else:
                        continue
                except Exception:
                    continue

                qb = self._extract_q_beta_from_track(trk)
                if qb is None:
                    continue
                q_t, beta = qb

                # 窗口推进
                buf = self.phys_buffers.get(trk.id)
                if buf is None:
                    buf = WindowBuffer(T=self.PHYS_T)
                    self.phys_buffers[trk.id] = buf
                buf.push(q_t, beta, feet_points_world=None)

                if buf.ready():
                    q_win, b, plane = buf.as_arrays()
                    try:
                        q_refined, forces = self.phys_wrapper.refine_window(q_win, b, plane)
                        q_last = q_refined[-1]
                        # 回写到 track 的状态（供下一帧用）
                        st = getattr(trk, "state", trk)
                        st.transl = q_last[0:3]
                        st.global_orient = q_last[3:6]
                        st.body_pose = q_last[6:6 + 69]
                        try:
                            st.forces = forces
                        except Exception:
                            pass
                    except Exception as e:
                        log.warning(f"[PhysPT] refine_window 失败（已跳过回写）：{e}")

            return out

        return wrapper


# =========================
#  Hydra 配置
# =========================
@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192, 256)
    pass


cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker with PhysPT integration."""
    tracker = HMR2_4dhuman(cfg)  # 我们在 __init__ 里已完成 PhysPT 钩子安装
    tracker.track()
    return None


if __name__ == "__main__":
    main()
