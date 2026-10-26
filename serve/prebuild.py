#!/usr/bin/env python3
"""Build the engine's CUDA extensions before the first serve, with no model loaded.

Run it once per container, with the engine's src/ on PYTHONPATH. Without it the server builds them lazily, after
~104 GiB of weights are in, and the compiler then needs several GiB the node no longer has. Calls each module's _ext().
usage: PYTHONPATH=<TENSORFOLD>/src python3 prebuild.py"""
import importlib
import time

import torch

torch.cuda.init()
MODS = ["tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.linear", "tensorfold.families.glm5_next.cuda.kda",
        "tensorfold.families.glm_moe_dsa.cuda.mla_pe", "tensorfold.families.glm5_next.cuda.qmm",
        "tensorfold.families.glm5_next.cuda.exl3_mm", "tensorfold.cuda.kernels.qmm", "tensorfold.cuda.experts"]
for name in MODS:
    t0 = time.time()
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[prebuild] {name}: import failed ({type(exc).__name__}: {exc})", flush=True)
        continue
    ext = getattr(mod, "_ext", None)
    if ext is None:
        print(f"[prebuild] {name}: no _ext", flush=True)
        continue
    try:
        ext()
        print(f"[prebuild] {name}: ready in {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[prebuild] {name}: build failed ({type(exc).__name__}: {str(exc)[:200]})", flush=True)
