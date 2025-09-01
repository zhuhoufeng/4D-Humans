# hmr2/physpt/adapter.py    （PhysPT 适配器，含短期外推与窗口精炼）
# -*- coding: utf-8 -*-
import os
import sys
import glob
import types
import importlib
import inspect
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix_list=("module.", "model.")):
    """Remove common prefixes in state_dict keys to allow flexible loading."""
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


def _find_model_entry(repo_dir: str):
    """
    Try to find a model constructor in third_party/PhysPT/models.
    Returns (module, builder_callable_or_class_name_str).
    """
    models_dir = os.path.join(repo_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"未找到目录: {models_dir}")

    # Ensure repo import path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Candidate module names to try first
    preferred = [
        "models.physpt_model",
        "models.model",
        "models.network",
        "models.physpt",
    ]

    # Candidate builders / classes
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

    # 1) Try preferred modules
    for mod_name in preferred:
        try:
            mod = importlib.import_module(mod_name)
            for bn in builder_names:
                if hasattr(mod, bn):
                    return mod, bn
        except Exception as e:
            last_err = e

    # 2) Fallback: import all files in models/ and search symbols
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

    # 3) If still not found, raise the last error for debugging
    raise ImportError(
        f"未能在 {models_dir} 中找到可用的模型构造函数/类。"
        f"请打开 PhysPT 的 models/ 下文件，确认存在如 build_model()/PhysPT 类，并在此处修改。\n"
        f"最后错误: {last_err}"
    )


def _build_model_from(mod: types.ModuleType, entry: str):
    """Instantiate model by entry; entry can be a function (builder) or a class name."""
    obj = getattr(mod, entry)
    if inspect.isfunction(obj):
        return obj()
    if inspect.isclass(obj):
        try:
            return obj()
        except TypeError:
            # 如果需要特定构造参数，可在这里自定义
            return obj
    # 万一是已实例化的对象
    return obj


class PhysPTWrapper:
    """
    统一封装 PhysPT 推理接口：
      - refine_window(q_window, beta, ground_plane) -> (q_refined, forces)
      - predict_next(q_window, beta, ground_plane, n_future=1) -> q_future  # 自回归外推

    输入/输出约定：
      q_window: (T, 75) = [transl(3), global_orient(axis-angle 3), body_pose(23*3)]
      beta:     (10,)
      ground_plane: Optional (4,) [nx, ny, nz, d], 表示平面 n^T x + d = 0
    """

    def __init__(self, physpt_repo_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.repo = os.path.abspath(physpt_repo_dir)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

        # 1) 自动搜寻权重
        assets_dir = os.path.join(self.repo, "assets")
        if not os.path.isdir(assets_dir):
            raise FileNotFoundError(
                f"未找到 PhysPT 资产目录: {assets_dir}\n"
                f"请先按 README 下载 assets，并覆盖到该目录。"
            )
        ckpt = None
        for pat in ("*.pth", "*.pt", "*.ckpt"):
            hits = glob.glob(os.path.join(assets_dir, pat))
            if hits:
                ckpt = hits[0]
                break
        if ckpt is None:
            raise FileNotFoundError(
                f"未在 {assets_dir} 找到 .pth/.pt/.ckpt 权重文件，请确认 assets 下载是否完成。"
            )

        # 2) 动态定位模型构造入口
        mod, entry = _find_model_entry(self.repo)
        self.model = _build_model_from(mod, entry)

        # 3) 加载权重（兼容不同 key 布局）
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix_if_present(state)
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            # 有些仓库把模型包了一层 .model
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                self.model.model.load_state_dict(state, strict=False)
            else:
                raise

        self.model.to(self.device).eval()

        # 可调阈值：当你在关联阶段构造 E_c（接触一致性）时可能会用到
        self.contact_threshold = 1.0

    # ---------- 公开 API ----------

    @torch.no_grad()
    def refine_window(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        对 (T,75) 的窗口做一次物理一致性精炼。
        返回:
          q_refined: (T,75) numpy
          forces:  可能包含 'lambda' (T, Nc*3), 'tau' (T,75), 'contacts'(T, Nc) 等
        """
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        # 根据不同实现尝试前向函数名
        outputs: Dict[str, Any] = {}
        # 1) 优先调用显式 refine
        if hasattr(self.model, "forward_refine"):
            outputs = self.model.forward_refine(q, b, plane)
        elif hasattr(self.model, "refine"):
            outputs = self.model.refine(q, b, plane)
        else:
            # 2) 兜底：有的实现直接 __call__
            out = self.model(q, b, plane)
            if isinstance(out, dict):
                outputs = out
            elif torch.is_tensor(out):
                outputs = {"q_refined": out}
            else:
                raise RuntimeError("PhysPT 模型未返回可识别的输出格式。")

        # 解析结果
        if "q_refined" in outputs:
            q_refined = outputs["q_refined"].detach().cpu().numpy()
            if q_refined.ndim == 3:  # (B,T,75)
                q_refined = q_refined[0]
        elif "q" in outputs:
            q_refined = outputs["q"].detach().cpu().numpy()
            if q_refined.ndim == 3:
                q_refined = q_refined[0]
        else:
            raise RuntimeError("PhysPT 输出里没有 'q_refined' 或 'q'。请检查模型接口。")

        forces = {}
        for k in ("lambda", "tau", "contacts"):
            if k in outputs and torch.is_tensor(outputs[k]):
                val = outputs[k].detach().cpu().numpy()
                if val.ndim >= 3:
                    val = val[0]  # 去掉 batch 维
                forces[k] = val

        return q_refined, forces

    @torch.no_grad()
    def predict_next(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None,
        n_future: int = 1
    ) -> np.ndarray:
        """
        用自回归的方式外推未来 n_future 帧。
        返回: (n_future, 75)
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
                out = self.model(cur, b, plane)  # 兜底

            if isinstance(out, dict) and "q_next" in out:
                q_next = out["q_next"]
            elif isinstance(out, dict) and "q_refined" in out:
                # 有些实现不显式给 next，就用 refine 后的最后一帧近似下一帧（保守策略）
                q_next = out["q_refined"][:, -1, :]
            elif torch.is_tensor(out):
                q_next = out[:, -1, :] if out.ndim == 3 else out
            else:
                raise RuntimeError("PhysPT 模型未返回 q_next / q_refined，可根据你的版本在此微调。")

            q_next = q_next.detach()
            preds.append(q_next.squeeze(0))

            # 自回归：把下一帧拼到窗口尾部，去掉最前一帧
            cur = torch.cat([cur[:, 1:, :], q_next[:, None, :]], dim=1)

        preds = torch.stack(preds, dim=0).detach().cpu().numpy()
        return preds
