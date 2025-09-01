# hmr2/physpt/adapter.py  —— 强化版
import os, sys, glob, types, importlib, inspect
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch

def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix_list=("module.", "model.")):
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

def _list_some_files(root, maxn=50):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.join(dp, f))
            if len(out) >= maxn:
                return out
    return out

def _find_model_entry(repo_dir: str):
    models_dir = os.path.join(repo_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"未找到目录: {models_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    preferred = ["models.physpt_model", "models.model", "models.network", "models.physpt"]
    builder_names = ["build_model", "get_model", "create_model", "PhysPT", "PhysPTModel", "Model", "Net"]
    last_err = None

    for mod_name in preferred:
        try:
            mod = importlib.import_module(mod_name)
            for bn in builder_names:
                if hasattr(mod, bn):
                    return mod, bn
        except Exception as e:
            last_err = e

    # fallback: brute-force import all
    for fname in os.listdir(models_dir):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"models.{fname[:-3]}")
            for bn in builder_names:
                if hasattr(mod, bn):
                    return mod, bn
        except Exception as e:
            last_err = e

    raise ImportError(f"未能在 {models_dir} 找到模型构造；最后错误: {last_err}")

def _build_model_from(mod: types.ModuleType, entry: str):
    obj = getattr(mod, entry)
    if inspect.isfunction(obj):
        return obj()
    if inspect.isclass(obj):
        try:
            return obj()
        except TypeError:
            return obj
    return obj

class PhysPTWrapper:
    def __init__(self, physpt_repo_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 允许用环境变量覆盖
        env_repo = os.environ.get("PHYSPT_DIR")
        if env_repo:
            physpt_repo_dir = env_repo

        self.repo = os.path.abspath(physpt_repo_dir)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

        # --- 1) 找 assets 目录（支持环境变量 PHYSPT_ASSETS；否则默认 <repo>/assets） ---
        assets_dir = os.environ.get("PHYSPT_ASSETS") or os.path.join(self.repo, "assets")
        assets_dir = os.path.abspath(assets_dir)

        # --- 2) 递归查找可识别后缀的权重 ---
        patterns = ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/*.pth.tar", "**/*.bin", "**/*.pkl"]
        candidates: List[str] = []
        if os.path.isdir(assets_dir):
            for pat in patterns:
                candidates += glob.glob(os.path.join(assets_dir, pat), recursive=True)

        if not candidates:
            # 打印帮助信息：列举assets里的部分文件，方便定位
            files_preview = _list_some_files(assets_dir) if os.path.isdir(assets_dir) else []
            hint = f"\n[PhysPT] assets路径: {assets_dir}\n[PhysPT] 发现文件(最多50条):\n  " + "\n  ".join(files_preview) if files_preview else f"\n[PhysPT] assets目录不存在: {assets_dir}"
            raise FileNotFoundError(
                "未在 assets 下找到权重文件（支持 .pth/.pt/.ckpt/.pth.tar/.bin/.pkl）。"
                "可设置环境变量 PHYSPT_ASSETS=<绝对路径> 或检查解压路径是否多了一层 assets。"
                + hint
            )

        ckpt = sorted(candidates, key=lambda p: (-os.path.getsize(p), p))[0]  # 选一个“看起来最大的”
        print(f"[PhysPT] 使用权重: {ckpt}")

        # --- 3) 构建模型并加载权重 ---
        mod, entry = _find_model_entry(self.repo)
        self.model = _build_model_from(mod, entry)

        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix_if_present(state)
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                self.model.model.load_state_dict(state, strict=False)
            else:
                raise

        self.model.to(self.device).eval()
        self.contact_threshold = 1.0

    @torch.no_grad()
    def refine_window(self, q_window: np.ndarray, beta: np.ndarray, ground_plane: Optional[np.ndarray] = None):
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        if hasattr(self.model, "forward_refine"):
            outputs = self.model.forward_refine(q, b, plane)
        elif hasattr(self.model, "refine"):
            outputs = self.model.refine(q, b, plane)
        else:
            out = self.model(q, b, plane)
            outputs = out if isinstance(out, dict) else {"q_refined": out}

        if "q_refined" in outputs:
            q_refined = outputs["q_refined"].detach().cpu().numpy()
            if q_refined.ndim == 3: q_refined = q_refined[0]
        elif "q" in outputs:
            q_refined = outputs["q"].detach().cpu().numpy()
            if q_refined.ndim == 3: q_refined = q_refined[0]
        else:
            raise RuntimeError("PhysPT 输出缺少 q_refined/q")

        forces = {}
        for k in ("lambda", "tau", "contacts"):
            if k in outputs and torch.is_tensor(outputs[k]):
                v = outputs[k].detach().cpu().numpy()
                if v.ndim >= 3: v = v[0]
                forces[k] = v
        return q_refined, forces

    @torch.no_grad()
    def predict_next(self, q_window: np.ndarray, beta: np.ndarray, ground_plane: Optional[np.ndarray] = None, n_future: int = 1):
        q = _safe_to_tensor(q_window, self.device)
        b = _safe_to_tensor(beta, self.device)
        plane = None if ground_plane is None else _safe_to_tensor(ground_plane, self.device)

        preds = []
        cur = q.clone()
        for _ in range(n_future):
            if hasattr(self.model, "forward_predict"):
                out = self.model.forward_predict(cur, b, plane)
            elif hasattr(self.model, "predict"):
                out = self.model.predict(cur, b, plane)
            else:
                out = self.model(cur, b, plane)

            if isinstance(out, dict) and "q_next" in out:
                q_next = out["q_next"]
            elif isinstance(out, dict) and "q_refined" in out:
                q_next = out["q_refined"][:, -1, :]
            elif torch.is_tensor(out):
                q_next = out[:, -1, :] if out.ndim == 3 else out
            else:
                raise RuntimeError("PhysPT 未返回 q_next/q_refined")

            q_next = q_next.detach()
            preds.append(q_next.squeeze(0))
            cur = torch.cat([cur[:, 1:, :], q_next[:, None, :]], dim=1)

        return torch.stack(preds, dim=0).cpu().numpy()
