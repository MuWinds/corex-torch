set -u
# 探针 125：逐元素算子（add/sub/mul/div）的 C++ 快路径——正确性（逐字节）+ p50 计时。
#
# 为什么做这个：probes/023 实测 `aten add` p50 78.8 µs（其中内核 28.7，其余是「aten 回调 +
# Python 实现体」），而 C++ 快路径的 mm 只要 3.1 µs。四个逐元素算子在 decode 每步出现 6–8 次，
# 是这条链上最值钱的一批。
#
# 判据（Lead 定的口径）：
#   * `aten add` **p50 ≤ 12 µs**（基线 78.8）；mul 标量同理；
#   * fp32 / fp16 / bf16 **三档都要过**，且逐元素**逐字节**比对（不是容差）；
#   * 覆盖 out 变体 / inplace 变体 / scalar 变体三条路径；
#   * inplace 的正确性必须在**独立副本**上验证（就地改，不能只看输出）。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/125_fast_ops.sh --name 125_fast_ops --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
export COREX_FAST_OPS=1
echo "HOST=$(hostname) DATE=$(date -Is) COREX_FAST_OPS=${COREX_FAST_OPS}"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/fast_ops.py" <<'PYEOF'
"""逐元素 C++ 快路径：三档 dtype × 三条路径，逐字节比对 + p50 计时。"""
import sys
import time

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _runtime as rt

corex_torch.register()
shim = corex_torch._device.shim()
copy_from = corex_torch._device._copy_from_impl

print("shim 导出数:", len([n for n in dir(shim) if not n.startswith("_")]))
print("C++ 接管的 schema:", shim.elementwise_op_names())
print("已登记逐元素内核 [op, variant, dtype, block, warps, n_ptrs, n_i32, n_f32, alpha_eff]:")
for row in shim.elementwise_kernel_info():
    print("   ", row)

ITERS = 500
DTYPES = [("fp32", torch.float32, 6), ("fp16", torch.float16, 5), ("bf16", torch.bfloat16, 15)]


def bench_p50(fn, iters=ITERS):
    fn()
    rt.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    rt.synchronize()
    samples.sort()
    return samples[len(samples) // 2]


def to_cpu_bytes(tensor):
    """按逻辑布局读回，返回字节串（逐字节比对用）。"""
    out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
    copy_from(tensor, out)
    return out.contiguous().view(torch.uint8).numpy().tobytes()


failures = []
timings = {}


def check(label, fn, expect_cpu, limit_us=None):
    try:
        fn()
        got = to_cpu_bytes(_RESULT[0])
        want = expect_cpu.contiguous().view(torch.uint8).numpy().tobytes()
        ok = got == want
        p50 = bench_p50(fn)
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL {label:44s} {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        failures.append(label)
        return
    timings[label] = p50
    over = limit_us is not None and p50 > limit_us
    verdict = "OK  " if ok and not over else "FAIL"
    note = "" if ok else "  ← 逐字节不符"
    if over:
        note += f"  ← 超过 {limit_us} us"
    limit = "" if limit_us is None else f" 门槛={limit_us}us"
    print(f"  {verdict} {label:44s} p50={p50:7.2f} us{limit}{note}")
    if not ok or over:
        failures.append(label)


#: inplace 用例要把结果放回这里给 check 读。
_RESULT = [None]

def check_pair(label, verify_fn, time_fn, expect_cpu, limit_us=None):
    """inplace 类：verify_fn 每次先重置再算一次（比数值），time_fn 纯计时（可以就地累加）。"""
    try:
        verify_fn()
        got = to_cpu_bytes(_RESULT[0])
        want = expect_cpu.contiguous().view(torch.uint8).numpy().tobytes()
        ok = got == want
        p50 = bench_p50(time_fn)
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL {label:44s} {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        failures.append(label)
        return
    timings[label] = p50
    over = limit_us is not None and p50 > limit_us
    verdict = "OK  " if ok and not over else "FAIL"
    note = "" if ok else "  ← 逐字节不符"
    if over:
        note += f"  ← 超过 {limit_us} us"
    limit = "" if limit_us is None else f" 门槛={limit_us}us"
    print(f"  {verdict} {label:44s} p50={p50:7.2f} us{limit}{note}")
    if not ok or over:
        failures.append(label)


N = 1024
for name, dtype, code in DTYPES:
    print(f"\n=== dtype={name}（c10 code {code}）===")
    a_cpu = (torch.arange(N, dtype=torch.float32) % 7 - 3).to(dtype) * 0.25
    b_cpu = (torch.arange(N, dtype=torch.float32) % 5 + 1).to(dtype) * 0.5
    a = a_cpu.to("corex")
    b = b_cpu.to("corex")
    out = torch.empty(N, dtype=dtype, device="corex")

    # out 变体（tensor）
    _RESULT[0] = out
    check(f"{name} add.out(tensor)", lambda: torch.add(a, b, out=out), a_cpu + b_cpu, 12)
    check(f"{name} add alpha=2 out", lambda: torch.add(a, b, alpha=2, out=out),
          a_cpu + 2.0 * b_cpu)
    check(f"{name} sub.out(tensor)", lambda: torch.sub(a, b, out=out), a_cpu - b_cpu)
    check(f"{name} mul.out(tensor)", lambda: torch.mul(a, b, out=out), a_cpu * b_cpu)
    safe_b = b_cpu.abs() + 0.5
    b2 = safe_b.to("corex")
    check(f"{name} div.out(tensor)", lambda: torch.div(a, b2, out=out), a_cpu / safe_b)

    # inplace（tensor）：每个用例用独立副本。**拷贝必须在计时之外**——
    # 之前把 copy_from 放进 lambda 里，量到的是「拷贝 + 算子」，会把 C++ 的收益算丢一半。
    view_t = torch.empty(N, dtype=dtype, device="corex")

    def inplace(op):
        op(view_t, b)                     # 就地改，不需要每轮重置（结果只用于最后一次比对）
        _RESULT[0] = view_t
        return view_t

    def _verify_tensor(op, expect):
        def run():
            copy_from(a, view_t)
            op(view_t, b)
            _RESULT[0] = view_t
        return run

    for op_name, op, expect in (("add_", torch.Tensor.add_, a_cpu + b_cpu),
                                ("sub_", torch.Tensor.sub_, a_cpu - b_cpu),
                                ("mul_", torch.Tensor.mul_, a_cpu * b_cpu)):
        verify = _verify_tensor(op, expect)
        check_pair(f"{name} {op_name}.Tensor", verify,
                   (lambda op=op: op(view_t, b)), expect, 12 if op_name == "add_" else None)

    # inplace（scalar）
    view_s = torch.empty(N, dtype=dtype, device="corex")

    def inplace_scalar(op, value):
        copy_from(a, view_s)
        _RESULT[0] = view_s
        return op(view_s, value)

    check(f"{name} add_.Scalar(+2.5)", lambda: inplace_scalar(torch.Tensor.add_, 2.5),
          a_cpu + 2.5, 12)
    check(f"{name} mul_.Scalar(*3)", lambda: inplace_scalar(torch.Tensor.mul_, 3.0), a_cpu * 3.0)
    check(f"{name} sub_.Scalar(-1.5)", lambda: inplace_scalar(torch.Tensor.sub_, 1.5),
          a_cpu - 1.5)

print("\n=== 结论 ===")
print(f"  用例数 {len(timings) + len(failures)}，失败 {len(failures)}"
      + (f"：{failures}" if failures else ""))
add_p50 = timings.get("fp32 add.out(tensor)")
if add_p50 is not None:
    print(f"  关键指标：fp32 add.out(tensor) p50 = {add_p50:.2f} us（基线 78.8，判据 ≤12）")
print("FAST_OPS_PASS" if not failures else "FAST_OPS_FAIL")
print("FAST_OPS_DONE")
raise SystemExit(0 if not failures else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/fast_ops.py" "$W/src"
RC=$?
echo "FAST_OPS_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE125_DONE $(date -Is)"
exit $RC
