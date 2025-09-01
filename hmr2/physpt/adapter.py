# hmr2/physpt/adapter.py
# -*- coding: utf-8 -*-
"""
PhysPT 适配器：
- 自动在 third_party/PhysPT (或 PHYSPT_DIR) 中定位模型代码与 assets
- 递归查找 assets 下的权重，并支持环境变量显式指定
- 灵活实例化模型（类/工厂函数/已实例对象），加载多样式 state_dict
- 提供 refine_window() 与 predict_next() 两个推理接口
"""

import os
import sys
import glob
import types
import importlib
import inspect
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch


# ----------------- 小工具 ----------------- #
def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor],
                             prefix_list=("module.", "model.", "net.", "network.")):
    """去除常见前缀，便于灵活加载权重。"""
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefix_list:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def _safe_to_tensor(x, device):
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t[None, ...]
    return t


def _list_some_files(root, maxn=50) -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.join(dp, f))
            if len(out) >= maxn:
                return out
    return out


# ----------------- 定位模型入口 ----------------- #
def _find_model_entry(repo_dir: str):
    """
    在 third_party/PhysPT/models/ 下寻找可用入口：
    - 优先模块：models.physpt_model / models.model / models.network / models.physpt
    - 入口名候选：build_model/get_model/create_model/PhysPT/PhysPTModel/Model/Net
    返回: (module, entry_name)
    """
    models_dir = os.path.join(repo_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"未找到目录: {models_dir}")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    preferred_modules = [
        "models.physpt_model",
        "models.model",
        "models.network",
        "models.physpt",
    ]
    builder_names = [
        "build_model",
        "get_model",
        "create_model",
        "PhysPT",
        "PhysPTModel",
        "Model",
        "Net",
    ]

    last_err = None

    # 1) 先试首选模块
    for mod_name in preferred_modules:
        try:
            mod = importlib.import_module(mod_name)
            for bn in builder_names:
                if hasattr(mod, bn):
                    return mod, bn
        except Exception as e:
            last_err = e

    # 2) 回退：扫描 models 目录所有 .py
    for fname in os.listdir(models_dir):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        mod_name = f"models.{fname[:-3]}"
        try:
            mod = importlib.import_module(mod_name)
            for bn in builder_names:
                if hasattr(mod, bn):
                    return mod, bn
        except Exception as e:
            last_err = e

    raise ImportError(
        f"未能在 {models_dir} 找到可用的模型构造函数/类；"
        f"请确认 PhysPT 的 models/ 下存在如 build_model()/PhysPT 类等入口。\n"
        f"最后错误：{last_err}"
    )


def _build_model_from(mod: types.ModuleType, entry: str):
    """entry 可能是函数(工厂)或类，返回其对象或类本身。"""
    obj = getattr(mod, entry)
    if inspect.isfunction(obj):
        return obj()
    if inspect.isclass(obj):
        # 注意：不在这里实例化，交由 _ensure_instance 统一处理
        return obj
    # 万一已经是实例
    return obj


def _ensure_instance(maybe_cls_or_obj):
    """
    若传入的是“类”，尽量实例化：
      1) 尝试无参构造；
      2) 尝试从 config.py 中获取默认配置 (Config / get_config / get_cfg)，再构造；
      3) 失败则抛异常，让用户手动指定构造参数。
    传入已是实例则原样返回。
    """
    if not inspect.isclass(maybe_cls_or_obj):
        return maybe_cls_or_obj

    cls = maybe_cls_or_obj

    # 1) 无参构造
    try:
        return cls()
    except TypeError:
        pass

    # 2) 使用 config.py
    try:
        cfg_mod = importlib.import_module("config")  # PhysPT 仓库根下通常有 config.py
        for name in ("Config", "get_config", "get_cfg"):
            if hasattr(cfg_mod, name):
                factory = getattr(cfg_mod, name)
                cfg = factory() if callable(factory) else factory
                try:
                    return cls(cfg)
                except Exception:
                    continue
    except Exception:
        pass

    # 3) 兜底
    raise RuntimeError(
        f"PhysPT 模型类 {cls.__name__} 需要构造参数，未能自动获取默认配置；"
        f"请在 adapter.py 中自定义对 {cls.__name__}(...) 的构造传参。"
    )


# ----------------- 主封装 ----------------- #
class PhysPTWrapper:
    """
    统一封装 PhysPT 推理接口：
      - refine_window(q_window, beta, ground_plane) -> (q_refined, forces)
      - predict_next(q_window, beta, ground_plane, n_future=1) -> q_future
    输入/输出约定：
      q_window: (T,75) = [transl(3), global_orient(axis-angle 3), body_pose(23*3)]
      beta:     (10,)
      ground_plane: Optional (4,) [nx, ny, nz, d], 表示平面 n^T x + d = 0
    """

    def __init__(self, physpt_repo_dir: str, device: str = "cuda"):
        # ---------- 设备 ----------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---------- 允许用环境变量覆盖根目录 ----------
        env_repo = os.environ.get("PHYSPT_DIR")
        if env_repo:
            physpt_repo_dir = env_repo

        self.repo = os.path.abspath(physpt_repo_dir)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

        # ---------- 定位 assets ----------
        assets_dir = os.environ.get("PHYSPT_ASSETS") or os.path.join(self.repo, "assets")
        assets_dir = os.path.abspath(assets_dir)

        # 递归查找候选权重
        patterns = ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/*.pth.tar", "**/*.bin", "**/*.pkl"]
        candidates: List[str] = []
        if os.path.isdir(assets_dir):
            for pat in patterns:
                candidates += glob.glob(os.path.join(assets_dir, pat), recursive=True)

        # 如果用户指定了确切 ckpt 路径，优先用它
        ckpt_env = os.environ.get("PHYSPT_CKPT")
        if ckpt_env and os.path.isfile(ckpt_env):
            ckpt = os.path.abspath(ckpt_env)
        else:
            if not candidates:
                # 打印 assets 下的部分文件帮助定位
                preview = _list_some_files(assets_dir, maxn=50)
                hint = f"\n[PhysPT] assets 路径: {assets_dir}"
                if preview:
                    hint += "\n[PhysPT] 发现文件(最多50条):\n  " + "\n  ".join(preview)
                else:
                    hint += "\n[PhysPT] assets 目录不存在或为空。"
                raise FileNotFoundError(
                    "未在 assets 下找到权重文件（支持 .pth/.pt/.ckpt/.pth.tar/.bin/.pkl）。"
                    "可设置环境变量 PHYSPT_ASSETS=<绝对路径> 或 PHYSPT_CKPT=<权重文件>。"
                    + hint
                )

            # 优先选择文件名包含 'physpt' 的候选；否则取体积最大者
            preferred = [p for p in candidates if "physpt" in os.path.basename(p).lower()]
            pick_from = preferred if preferred else candidates
            ckpt = sorted(pick_from, key=lambda p: (-os.path.getsize(p), p))[0]

        print(f"[PhysPT] 使用权重: {ckpt}")

        # ---------- 构建模型 ----------
        mod, entry = _find_model_entry(self.repo)
        model_obj_or_cls = _build_model_from(mod, entry)
        self.model = _ensure_instance(model_obj_or_cls)

        # ---------- 加载权重 ----------
        state = torch.load(ckpt, map_location="cpu")

        # 兼容多种保存格式：优先取常见键，否则若直接就是 state_dict 也 OK
        if isinstance(state, dict):
            for k in ("state_dict", "model", "model_state_dict", "net", "network", "weights", "params"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break

        if isinstance(state, dict):
            state = _strip_prefix_if_present(state)

        loaded = False
        try:
            self.model.load_state_dict(state, strict=False)
            loaded = True
        except Exception:
            # 有些工程真正的 nn.Module 在 .model
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                self.model.model.load_state_dict(state, strict=False)
                loaded = True

        if not loaded:
            raise RuntimeError("无法将 checkpoint 加载到 PhysPT 模型，请检查模型类与权重是否匹配。")

        self.model.to(self.device).eval()

        # 可调阈值：关联阶段若构建 E_c（接触一致性）可用
        self.contact_threshold = 1.0

    # ----------------- 推理：窗口精炼 ----------------- #
    @torch.no_grad()
    def refine_window(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        对 (T,75) 的窗口做物理一致性精炼。
        返回:
          q_refined: (T,75) numpy
          forces:  可能包含 'lambda' (T, Nc*3), 'tau' (T,75), 'contacts'(T, Nc) 等
        """
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        # 根据不同实现尝试前向函数名
        outputs: Dict[str, Any] = {}
        if hasattr(self.model, "forward_refine"):
            outputs = self.model.forward_refine(q, b, plane)
        elif hasattr(self.model, "refine"):
            outputs = self.model.refine(q, b, plane)
        else:
            out = self.model(q, b, plane)
            outputs = out if isinstance(out, dict) else {"q_refined": out}

        # 解析结果
        if "q_refined" in outputs and torch.is_tensor(outputs["q_refined"]):
            q_refined = outputs["q_refined"].detach().cpu().numpy()
            if q_refined.ndim == 3:  # (B,T,75)
                q_refined = q_refined[0]
        elif "q" in outputs and torch.is_tensor(outputs["q"]):
            q_refined = outputs["q"].detach().cpu().numpy()
            if q_refined.ndim == 3:
                q_refined = q_refined[0]
        else:
            raise RuntimeError("PhysPT 输出缺少 'q_refined' 或 'q' 张量。")

        forces: Dict[str, np.ndarray] = {}
        for k in ("lambda", "tau", "contacts"):
            if k in outputs and torch.is_tensor(outputs[k]):
                v = outputs[k].detach().cpu().numpy()
                if v.ndim >= 3:
                    v = v[0]
                forces[k] = v

        return q_refined, forces

    # ----------------- 推理：自回归外推未来 n 帧 ----------------- #
    @torch.no_grad()
    def predict_next(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None,
        n_future: int = 1
    ) -> np.ndarray:
        """
        自回归方式外推未来 n_future 帧，返回 (n_future, 75) numpy。
        """
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        preds: List[torch.Tensor] = []
        cur = q.clone()

        for _ in range(n_future):
            if hasattr(self.model, "forward_predict"):
                out = self.model.forward_predict(cur, b, plane)
            elif hasattr(self.model, "predict"):
                out = self.model.predict(cur, b, plane)
            else:
                out = self.model(cur, b, plane)

            if isinstance(out, dict) and "q_next" in out and torch.is_tensor(out["q_next"]):
                q_next = out["q_next"]
            elif isinstance(out, dict) and "q_refined" in out and torch.is_tensor(out["q_refined"]):
                q_next = out["q_refined"][:, -1, :]
            elif torch.is_tensor(out):
                q_next = out[:, -1, :] if out.ndim == 3 else out
            else:
                raise RuntimeError("PhysPT 推理未返回 'q_next' 或 'q_refined' 可用于外推。")

            q_next = q_next.detach()
            preds.append(q_next.squeeze(0))

            # 自回归滚动
            cur = torch.cat([cur[:, 1:, :], q_next[:, None, :]], dim=1)

        return torch.stack(preds, dim=0).cpu().numpy()
