set -u
# 探针 126：为什么 `_aten_kernels._register_fast_ops()` 一个内核都没登记？（定位 + 直接验证 C++）
#
# 现象（probes/125）：`shim.elementwise_kernel_info()` 是**空表**，于是
#   * `add_.Tensor` 走到 C++ 就报「还没登记内核」；
#   * `add.out` 反而能用且 p50≈44 µs —— 因为 `_device` 把 add.out 注册成了 Python 实现
#     （probes/125 里那条 `Overriding ... aten::add.out` 警告就是它），C++ 实现被盖住了。
#
# 本探针按顺序回答三件事：
#   1. 内核组给的两个入口（`fast_variants()` / `get_launch_meta()`）在当前部署里能不能用；
#   2. `_register_fast_ops()` 内部那个 `except: continue` 吞掉的到底是哪个异常；
#   3. 手动按同一份 meta 调 `shim.load_fast_op()` 之后，C++ 的 `add_.Tensor` 是否
#      逐字节正确、p50 是多少（这条路径没被 Python 覆盖，能直接量到 C++）。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/126_fast_op_diag.sh --name 126_fast_op_diag --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
export COREX_FAST_OPS=1
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/fast_op_diag.py" <<'PYEOF'
"""定位注册链断点，并直接验证 C++ 逐元素路径。"""
import sys
import time
import traceback

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _ops_elementwise as elt
from corex_torch import _runtime as rt

corex_torch.register()
shim = corex_torch._device.shim()
copy_from = corex_torch._device._copy_from_impl

print("=== 1. 内核组入口是否可用 ===")
print(f"  fast_variants: {hasattr(elt, 'fast_variants')}")
print(f"  get_launch_meta: {hasattr(elt, 'get_launch_meta')}")
print(f"  FASTPATH_OPS: {getattr(elt, 'FASTPATH_OPS', None)}")
if hasattr(elt, "fast_variants"):
    print(f"  fast_variants() = {elt.fast_variants()}")
print(f"  注册后 elementwise_kernel_info() = {shim.elementwise_kernel_info()}")

print("\n=== 2. 让异常现形：逐个 (op, variant, dtype) 试 get_launch_meta ===")
ok_metas = {}
if hasattr(elt, "fast_variants") and hasattr(elt, "get_launch_meta"):
    for op, variant in elt.fast_variants():
        if op not in getattr(elt, "FASTPATH_OPS", ()):
            continue
        for dtype in ("fp32", "fp16", "bf16"):
            try:
                meta = elt.get_launch_meta(op, dtype, variant)
                ok_metas[(op, variant, dtype)] = meta
                if dtype == "fp32":
                    keys = sorted(meta.keys())
                    print(f"  OK   {op}/{variant}/{dtype}  keys={keys}")
            except Exception as exc:                    # noqa: BLE001
                print(f"  FAIL {op}/{variant}/{dtype}  {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:110]}")
                if dtype == "fp32":
                    traceback.print_exc(limit=3)
print(f"  可用的 meta 组合数: {len(ok_metas)}")

print("\n=== 3. 手动按同一份 meta 登记，然后量 C++ 的 add_.Tensor ===")
if ok_metas:
    for (op, variant, dtype), meta in ok_metas.items():
        try:
            shim.load_fast_op(op, variant, dtype, meta["cubin"], int(meta["num_warps"]),
                              int(meta["num_stages"]), int(meta["n_ptrs"]), int(meta["n_i32"]),
                              int(meta["n_f32"]), bool(meta["alpha_effective"]),
                              int(meta["block"]), meta["name"], int(meta.get("shared", 0)))
        except Exception as exc:                        # noqa: BLE001
            print(f"  load_fast_op 失败 {op}/{variant}/{dtype}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}")
    print("  登记后 elementwise_kernel_info():")
    for row in shim.elementwise_kernel_info():
        print("   ", row)

    ITERS = 500
    N = 1024
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16),
                              ("bf16", torch.bfloat16)):
        a_cpu = (torch.arange(N, dtype=torch.float32) % 7 - 3).to(dtype) * 0.25
        b_cpu = (torch.arange(N, dtype=torch.float32) % 5 + 1).to(dtype) * 0.5
        a, b = a_cpu.to("corex"), b_cpu.to("corex")
        view = torch.empty(N, dtype=dtype, device="corex")

        def run():
            copy_from(a, view)
            return view.add_(b)

        try:
            run()
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAIL {dtype_name} add_.Tensor  {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}")
            continue
        back = torch.empty(N, dtype=dtype)
        copy_from(view, back)
        want = a_cpu + b_cpu
        same = back.view(torch.uint8).numpy().tobytes() == want.view(torch.uint8).numpy().tobytes()
        rt.synchronize()
        samples = []
        for _ in range(ITERS):
            started = time.perf_counter()
            run()
            samples.append((time.perf_counter() - started) * 1e6)
        rt.synchronize()
        samples.sort()
        p50 = samples[len(samples) // 2]
        print(f"  {'OK  ' if same else 'FAIL'} {dtype_name} add_.Tensor（C++ 路径）"
              f"  逐字节一致={same}  p50={p50:7.2f} us")
else:
    print("  没有可用 meta，跳过")

print("\n=== 3.5 不传 name（只靠候选名兜底）能否成功 ===")
meta = ok_metas.get(("mul", "tensor", "fp32"))
if meta is not None:
    try:
        shim.load_fast_op("mul", "tensor", "fp32", meta["cubin"], int(meta["num_warps"]),
                          int(meta["num_stages"]), int(meta["n_ptrs"]), int(meta["n_i32"]),
                          int(meta["n_f32"]), bool(meta["alpha_effective"]), int(meta["block"]),
                          "", int(meta.get("shared", 0)))
        print(f"  OK   不传 name 也装载成功（候选名兜底有效）meta['name']={meta['name']}")
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL 不传 name 装载失败：{type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        print(f"       → 需要 Python 侧把 meta['name']={meta['name']!r} 传进来")
print("\n=== 4. 结论 ===")
print(f"  meta 可用组合 {len(ok_metas)}/24")
print("FAST_OP_DIAG_DONE")
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/fast_op_diag.py" "$W/src"
RC=$?
echo "FAST_OP_DIAG_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE126_DONE $(date -Is)"
exit $RC
