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
        mod_name = f"models.{f_
