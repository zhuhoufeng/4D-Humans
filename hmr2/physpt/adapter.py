# hmr2/physpt/adapter.py
# -*- coding: utf-8 -*-
"""
PhysPT 适配器（稳健版）：
- 递归搜索 assets/** 权重；支持 PHYSPT_DIR / PHYSPT_ASSETS / PHYSPT_CKPT
- 动态按文件路径加载 PhysPT 的 models/*.py（即使没有 __init__.py）
- 自动读取 config.py / constants.py 提取默认参数
- 实例化时按 PhysPT(device, seqlen, mode, f_dim, d_model, nhead, nlayers) 的“7 个位置参数”严格传参；
  同时也尝试 kwargs 与 torch.device/字符串两种设备形式。
- 兼容多种 state_dict 键名并去前缀
- 提供 refine_window() / predict_next() 两个推理接口
"""

import os
import sys
import glob
import types
import importlib
import importlib.util
import inspect
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch


# =============== 基础工具 ===============

def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor],
                             prefix_list=("module.", "model.", "net.", "network.")):
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


# =============== 动态加载/读取配置 ===============

def _load_module_from_path(py_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {py_path} 构建导入 spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _gather_candidates(repo_dir: str) -> List[types.ModuleType]:
    mods: List[types.ModuleType] = []
    search_dirs = [
        os.path.join(repo_dir, "models"),
        repo_dir,
    ]
    counter = 0
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py") or fn.startswith("_"):
                    continue
                path = os.path.join(root, fn)
                base = fn.lower()
                if ("physpt" in base) or ("model" in base) or ("network" in base) or ("net" in base):
                    counter += 1
                    mod_name = f"physpt_dyn_{counter}"
                    try:
                        mods.append(_load_module_from_path(path, mod_name))
                    except Exception:
                        pass
    return mods


def _load_cfg_and_consts(repo_dir: str):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    cfg_dict: Dict[str, Any] = {}
    const_dict: Dict[str, Any] = {}

    # config.py
    try:
        cfg_mod = importlib.import_module("config")
        for n in dir(cfg_mod):
            if n.startswith("_"): 
                continue
            obj = getattr(cfg_mod, n)
            if callable(obj):
                try:
                    val = obj()
                    cfg_dict[n.lower()] = val
                except TypeError:
                    pass
                except Exception:
                    pass
            else:
                cfg_dict[n.lower()] = obj
    except Exception:
        pass

    # constants.py
    try:
        c_mod = importlib.import_module("constants")
        for n in dir(c_mod):
            if n.startswith("_"): 
                continue
            const_dict[n.lower()] = getattr(c_mod, n)
    except Exception:
        pass

    return cfg_dict, const_dict


def _find_entry_symbol(mods: List[types.ModuleType]):
    # 1) 类名包含 PhysPT
    for m in mods:
        for n, obj in vars(m).items():
            if inspect.isclass(obj) and "physpt" in n.lower():
                return obj
    # 2) 工厂函数
    for m in mods:
        for fname in ("build_model", "get_model", "create_model"):
            if hasattr(m, fname) and callable(getattr(m, fname)):
                return getattr(m, fname)
    # 3) 备用：Model/Net
    for m in mods:
        for n, obj in vars(m).items():
            if inspect.isclass(obj) and n.lower() in ("model", "net"):
                return obj
    raise ImportError("未找到 PhysPT 入口（类名包含 'PhysPT' 或 build_model/get_model/create_model）。")


# =============== 构造入口（关键部分：7 个位置参数） ===============

def _construct_entry(entry, device, cfg_dict, const_dict):
    """
    直接按 PhysPT(device, seqlen, mode, f_dim, d_model, nhead, nlayers) 构造；
    1) kwargs + 字符串设备
    2) kwargs + torch.device
    3) 位置参数 + 字符串设备
    4) 位置参数 + torch.device
    """
    # 设备字符串
    if isinstance(device, torch.device):
        device_str = device.type
    elif isinstance(device, str):
        device_str = 'cuda' if 'cuda' in device else 'cpu'
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 从 config/constants 拿参数（取不到就用默认值）
    seqlen  = int(cfg_dict.get('seqlen', 16))
    mode    = str(cfg_dict.get('mode', 'inference'))
    f_dim   = int(const_dict.get('f_dim', 75))
    d_model = int(const_dict.get('d_model', const_dict.get('hidden_dim', 256)))
    nhead   = int(const_dict.get('nhead', const_dict.get('num_heads', 8)))
    nlayers = int(const_dict.get('nlayers', const_dict.get('num_layers', 6)))

    # ——— 1) kwargs + 字符串设备 ———
    try:
        return entry(device=device_str, seqlen=seqlen, mode=mode,
                     f_dim=f_dim, d_model=d_model, nhead=nhead, nlayers=nlayers)
    except Exception as e_kw_str:
        last = e_kw_str

    # ——— 2) kwargs + torch.device ———
    try:
        return entry(device=torch.device(device_str), seqlen=seqlen, mode=mode,
                     f_dim=f_dim, d_model=d_model, nhead=nhead, nlayers=nlayers)
    except Exception as e_kw_dev:
        last = e_kw_dev

    # ——— 3) 位置参数 + 字符串设备（严格 7 个）———
    try:
        return entry(device_str, seqlen, mode, f_dim, d_model, nhead, nlayers)
    except Exception as e_pos_str:
        last = e_pos_str

    # ——— 4) 位置参数 + torch.device ———
    try:
        return entry(torch.device(device_str), seqlen, mode, f_dim, d_model, nhead, nlayers)
    except Exception as e_pos_dev:
        last = e_pos_dev

    raise RuntimeError(f"无法实例化/构造 PhysPT 模型；最后错误：{last}")


# =============== 主封装 ===============

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
        # 设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 环境变量覆盖根目录
        env_repo = os.environ.get("PHYSPT_DIR")
        if env_repo:
            physpt_repo_dir = env_repo

        self.repo = os.path.abspath(physpt_repo_dir)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

        # assets
        assets_dir = os.environ.get("PHYSPT_ASSETS") or os.path.join(self.repo, "assets")
        assets_dir = os.path.abspath(assets_dir)

        # 递归找权重
        patterns = ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/*.pth.tar", "**/*.bin", "**/*.pkl"]
        candidates: List[str] = []
        if os.path.isdir(assets_dir):
            for pat in patterns:
                candidates += glob.glob(os.path.join(assets_dir, pat), recursive=True)

        ckpt_env = os.environ.get("PHYSPT_CKPT")
        if ckpt_env and os.path.isfile(ckpt_env):
            ckpt = os.path.abspath(ckpt_env)
        else:
            if not candidates:
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
            preferred = [p for p in candidates if "physpt" in os.path.basename(p).lower()]
            pick_from = preferred if preferred else candidates
            ckpt = sorted(pick_from, key=lambda p: (-os.path.getsize(p), p))[0]

        print(f"[PhysPT] 使用权重: {ckpt}")

        # 构建模型（动态加载 + 7 位置参数兜底）
        cfg_dict, const_dict = _load_cfg_and_consts(self.repo)
        mods = _gather_candidates(self.repo)
        entry = _find_entry_symbol(mods)
        self.model = _construct_entry(entry, self.device, cfg_dict, const_dict)

        # 加载权重
        state = torch.load(ckpt, map_location="cpu")
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
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                self.model.model.load_state_dict(state, strict=False)
                loaded = True
        if not loaded:
            raise RuntimeError("无法将 checkpoint 加载到 PhysPT 模型，请检查模型类与权重是否匹配。")

        self.model.to(self.device).eval()
        self.contact_threshold = 1.0

    # ------- 推理：窗口精炼 ------- #
    @torch.no_grad()
    def refine_window(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        outputs: Dict[str, Any] = {}
        if hasattr(self.model, "forward_refine"):
            outputs = self.model.forward_refine(q, b, plane)
        elif hasattr(self.model, "refine"):
            outputs = self.model.refine(q, b, plane)
        else:
            out = self.model(q, b, plane)
            outputs = out if isinstance(out, dict) else {"q_refined": out}

        if "q_refined" in outputs and torch.is_tensor(outputs["q_refined"]):
            q_refined = outputs["q_refined"].detach().cpu().numpy()
            if q_refined.ndim == 3:
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

    # ------- 推理：自回归外推 ------- #
    @torch.no_grad()
    def predict_next(
        self,
        q_window: np.ndarray,
        beta: np.ndarray,
        ground_plane: Optional[np.ndarray] = None,
        n_future: int = 1
    ) -> np.ndarray:
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
                raise RuntimeError("PhysPT 推理未返回 'q_next' 或 'q_refined'。")

            q_next = q_next.detach()
            preds.append(q_next.squeeze(0))
            cur = torch.cat([cur[:, 1:, :], q_next[:, None, :]], dim=1)

        return torch.stack(preds, dim=0).cpu().numpy()
