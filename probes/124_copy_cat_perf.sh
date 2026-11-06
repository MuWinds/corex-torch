set -u
# 探针 124：搬运路径（copy_ / cat）的 p50 计时与正确性。
#
# 为什么做这个：probes/022 的逐步 decode 17.1 ms/步、约 31 个算子（≈550 µs/算子），
# 而发射地板只有 10.8 µs。Lead 用 p50 + 500 次把范围锁到了 `_copy_impl` 的**非连续源**：
# `[4,6,16].transpose(1,2).contiguous()` 约 2 ms（≈5 µs/元素），因为旧实现只找「目的后缀
# 连续段」，一转置就退化成逐元素，每个元素付一次 Python + 同步 memcpy 的固定成本。
#
# 本探针的判据（Lead 定的口径）：
#   * 统计量一律 **p50**，迭代 **500** 次；
#   * 带两条基线（空循环 / 只分配 64 元素），用来划清「测试框架本身的地板」；
#   * 三个关键形状给出明确阈值：转置 384 元素 ≤400 µs（终态目标 ≤100）、
#     `[1,64][...,0::2]`（32 元素）≤60 µs；cat 不许退化；
#   * 每条都同时校验与 CPU 逐元素一致——性能口径不能替代正确性。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/124_copy_cat_perf.sh --name 124_copy_cat_perf --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/copy_cat_perf.py" <<'PYEOF'
"""p50 计时（500 次）+ 两条基线 + 逐条正确性校验。"""
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
DT = corex_torch._device._scalar_type(torch.float32)

ITERS = 500


def bench_p50(fn, iters=ITERS):
    """返回 p50（微秒）：迭代内各自取时间，循环外只同步一次。"""
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


def readback(tensor):
    out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
    copy_from(tensor, out)
    return out


results = []


def case(label, fn, want_cpu, threshold_us, hint):
    try:
        value = fn()
        ok = torch.equal(readback(value), want_cpu)
        p50 = bench_p50(fn)
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL {label:38s} {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        results.append((label, None, False, threshold_us))
        return
    verdict = "OK  " if ok and (threshold_us is None or p50 <= threshold_us) else "FAIL"
    limit = "—" if threshold_us is None else f"{threshold_us} us"
    print(f"  {verdict} {label:38s} p50={p50:8.2f} us  门槛={limit:>8s}  {hint}")
    results.append((label, p50, ok, threshold_us))


print("=== 1. 基线（把「测试框架地板」和「一次分配的代价」划出来）===")
base_empty = bench_p50(lambda: None)
base_fill = bench_p50(lambda: shim.empty_strided([64], [1], DT, 0))
print(f"  --   空循环基线                            p50={base_empty:8.2f} us")
print(f"  --   只分配 64 元素                         p50={base_fill:8.2f} us")

print("\n=== 2. Lead 锁定的三个形状（p50，500 次）===")
t_cpu = torch.arange(4 * 6 * 16, dtype=torch.float32).reshape(4, 6, 16) * 0.5
t = t_cpu.to("corex")
case("[4,6,16].transpose(1,2).contiguous()", lambda: t.transpose(1, 2).contiguous(),
     t_cpu.transpose(1, 2).contiguous(), 400, "384 元素；旧实现 ≈2000 us")

x_cpu = torch.arange(64, dtype=torch.float32).reshape(1, 64) * 0.25
x = x_cpu.to("corex")
case("[1,64][...,0::2].contiguous()", lambda: x[..., 0::2].contiguous(),
     x_cpu[..., 0::2].contiguous(), 60, "32 元素；旧实现 ≈156 us")

a_cpu = torch.arange(4 * 6 * 16, dtype=torch.float32).reshape(4, 6, 16) * 0.5
b_cpu = torch.arange(4 * 1 * 16, dtype=torch.float32).reshape(4, 1, 16) + 1.0
a, b = a_cpu.to("corex"), b_cpu.to("corex")
cat_out = torch.empty(4, 7, 16, device="corex")
case("cat [4,6,16]+[4,1,16] dim=1", lambda: torch.cat((a, b), dim=1, out=cat_out),
     torch.cat((a_cpu, b_cpu), dim=1), 150, "384 元素；Lead 测到 117 us")

print("\n=== 3. 三档大小（copy_ / cat），看是否随字节数线性 ===")
TIERS = [("1 KiB", 256), ("64 KiB", 16384), ("1 MiB", 262144)]
for tier_name, nelem in TIERS:
    src_cpu = torch.arange(nelem, dtype=torch.float32)
    src = src_cpu.to("corex")
    dst = torch.empty(nelem, device="corex")
    case(f"{tier_name} copy_ 连续源", lambda: dst.copy_(src), src_cpu, None, "1 条 memcpy")

    batch, inner = 4, max(1, nelem // 4)
    one_cpu = torch.arange(inner, dtype=torch.float32).reshape(1, inner)
    one = one_cpu.to("corex")
    expanded = one.expand(batch, inner)
    dst3 = torch.empty(batch, inner, device="corex")
    case(f"{tier_name} copy_ 广播源（expand）", lambda: dst3.copy_(expanded),
         one_cpu.expand(batch, inner).contiguous(), None, f"{batch} 条 memcpy")

    half = nelem // 2
    p_cpu = torch.arange(half, dtype=torch.float32)
    q_cpu = torch.arange(half, dtype=torch.float32) + 1.0
    p, q = p_cpu.to("corex"), q_cpu.to("corex")
    out0 = torch.empty(nelem, device="corex")
    case(f"{tier_name} cat dim=0", lambda: torch.cat((p, q), dim=0, out=out0),
         torch.cat((p_cpu, q_cpu), dim=0), None, "2 条 memcpy")

print("\n=== 4. 回归：probes/022 里被旧检查拒掉的 [4,1,16] stride=(16,64,1) ===")
flat = torch.arange(64, dtype=torch.float32)
ref_cpu = torch.as_strided(flat, (4, 1, 16), (16, 64, 1))
base = flat.to("corex")
sliced = torch.as_strided(base, (4, 1, 16), (16, 64, 1))
tail_cpu = torch.arange(4 * 3 * 16, dtype=torch.float32).reshape(4, 3, 16) + 1.0
tail = tail_cpu.to("corex")
out = torch.empty(4, 4, 16, device="corex")
print(f"  输入 is_contiguous()={sliced.is_contiguous()} stride={tuple(sliced.stride())}")
case("cat([4,1,16]+[4,3,16], dim=1)", lambda: torch.cat((sliced, tail), dim=1, out=out),
     torch.cat((ref_cpu, tail_cpu), dim=1), None, "size-1 维不约束 stride")

print("\n=== 5. 结论 ===")
bad = [item for item in results
       if item[1] is None or not item[2]
       or (item[3] is not None and item[1] > item[3])]
print(f"  用例数 {len(results)}，未达标 {len(bad)}" + (f"：{[x[0] for x in bad]}" if bad else ""))
print(f"  基线：空循环 {base_empty:.2f} us / 分配 64 元素 {base_fill:.2f} us")
print("COPY_CAT_PERF_PASS" if not bad else "COPY_CAT_PERF_FAIL")
print("COPY_CAT_PERF_DONE")
raise SystemExit(0 if not bad else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/copy_cat_perf.py" "$W/src"
RC=$?
echo "COPY_CAT_PERF_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE124_DONE $(date -Is)"
exit $RC
