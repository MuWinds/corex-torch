set -u
# 探针 124：cat / copy_ 的「连续情形单次 memcpy」快路径，三档大小的耗时。
#
# 为什么做这个：probes/022 的逐步 decode 是 17.1 ms/步、约 31 个算子（≈550 µs/算子），
# 而发射地板只有 10.8 µs——差两个数量级，问题不在内核而在「Python 级分段搬」。
# 本探针把三类搬运各量三档（1 KiB / 64 KiB / 1 MiB）：
#   * copy_ 连续源     → 期望 1 次 memcpy
#   * copy_ 转置视图   → 「末尾连续段 + 外层 stride 走」的通用路径
#   * copy_ 广播视图   → stride==0 的前导维（expand 的典型形态，之前会退化成逐元素）
#   * cat dim=0        → 每个输入 1 次
#   * cat 中间维（KV 形态 [4, L, 16]）→ outer × 输入数 次
#   * contiguous(expand) → empty_like + copy_
# 同时校验每条的结果与 CPU 一致（性能测试不能替代正确性），并复验 probes/022 里
# `[4, 1, 16]` stride=(16, 64, 1) 那个「torch 语义下连续、却被旧检查拒掉」的输入。
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
"""量搬运路径的三档耗时，并逐条校验正确性。"""
import sys
import time

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _runtime as rt

corex_torch.register()
copy_from = corex_torch._device._copy_from_impl

TIERS = [("1 KiB", 256), ("64 KiB", 16384), ("1 MiB", 262144)]   # fp32 元素数
ITERS = 200


def bench(fn) -> float:
    """循环外同步一次，返回单次平均微秒。"""
    fn()
    rt.synchronize()
    started = time.perf_counter()
    for _ in range(ITERS):
        fn()
    rt.synchronize()
    return (time.perf_counter() - started) / ITERS * 1e6


def readback(tensor):
    """把 corex 张量按逻辑布局读回 CPU。"""
    out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
    copy_from(tensor, out)
    return out


def report(label, fn, bytes_moved, want_cpu, call_hint):
    """跑基准 + 校验 + 打一行。"""
    try:
        us = bench(fn)
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL {label:34s} {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        return None
    got = readback(fn())
    ok = torch.equal(got, want_cpu)
    gbs = bytes_moved / (us * 1e-6) / 1e9 if us > 0 else 0.0
    flag = "OK  " if ok else "FAIL"
    print(f"  {flag} {label:34s} {us:8.2f} us  {gbs:7.1f} GB/s  memcpy≈{call_hint}"
          + ("" if ok else "  ← 数值不符"))
    return us if ok else None


def main():
    results = {}
    for tier_name, nelem in TIERS:
        nbytes = nelem * 4
        print(f"\n=== {tier_name}（{nelem} 个 fp32）===")

        # --- copy_ 连续源 ---
        src_cpu = torch.arange(nelem, dtype=torch.float32)
        src = src_cpu.to("corex")
        dst = torch.empty(nelem, device="corex")
        results[(tier_name, "copy_contig")] = report(
            "copy_ 连续源", lambda: dst.copy_(src), 2 * nbytes, src_cpu, 1)

        # --- copy_ 转置视图（二维） ---
        cols = 64
        rows = max(1, nelem // cols)
        base_cpu = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        base = base_cpu.to("corex")
        view = base.t()                                  # 非连续
        dst2 = torch.empty(cols, rows, device="corex")
        results[(tier_name, "copy_transposed")] = report(
            "copy_ 转置视图", lambda: dst2.copy_(view), 2 * nbytes, base_cpu.t(),
            f"≈{rows}（每行一段）")

        # --- copy_ 广播视图（前导 stride==0） ---
        batch = 4
        inner = max(1, nelem // batch)
        one_cpu = torch.arange(inner, dtype=torch.float32).reshape(1, inner)
        one = one_cpu.to("corex")
        expanded = one.expand(batch, inner)
        dst3 = torch.empty(batch, inner, device="corex")
        results[(tier_name, "copy_expanded")] = report(
            "copy_ 广播视图（expand）", lambda: dst3.copy_(expanded), 2 * nbytes,
            one_cpu.expand(batch, inner).contiguous(), batch)

        # --- contiguous(expand) ---
        results[(tier_name, "contiguous_expanded")] = report(
            "contiguous() 于 expand 结果", lambda: expanded.contiguous(), 2 * nbytes,
            one_cpu.expand(batch, inner).contiguous(), batch)

        # --- cat dim=0 ---
        half = nelem // 2
        a_cpu = torch.arange(half, dtype=torch.float32)
        b_cpu = torch.arange(half, dtype=torch.float32) + 1.0
        a, b = a_cpu.to("corex"), b_cpu.to("corex")
        out0 = torch.empty(nelem, device="corex")
        results[(tier_name, "cat_dim0")] = report(
            "cat dim=0（两个输入）", lambda: torch.cat((a, b), dim=0, out=out0), 2 * nbytes,
            torch.cat((a_cpu, b_cpu), dim=0), 2)

        # --- cat 中间维（KV cache 形态：[4, L, 16] 沿 dim=1 拼） ---
        heads = 16
        length = max(1, nelem // (4 * heads) // 2)
        k_cpu = torch.arange(4 * length * heads, dtype=torch.float32).reshape(4, length, heads)
        kv_cpu = torch.arange(4 * length * heads, dtype=torch.float32).reshape(4, length, heads) + 1.0
        k, kv = k_cpu.to("corex"), kv_cpu.to("corex")
        out1 = torch.empty(4, 2 * length, heads, device="corex")
        results[(tier_name, "cat_kv")] = report(
            "cat 中间维 [4,L,16]（KV 形态）",
            lambda: torch.cat((k, kv), dim=1, out=out1), 2 * nbytes,
            torch.cat((k_cpu, kv_cpu), dim=1), 4 * 2)

    # --- 复验 probes/022 报的那个 stride：(16, 64, 1) 的 [4, 1, 16] ---
    print("\n=== 回归：probes/022 里被旧检查拒掉的 [4,1,16] stride=(16,64,1) ===")
    base_cpu = torch.arange(4 * 4 * 16, dtype=torch.float32).reshape(4, 4, 16)
    base = base_cpu.to("corex")
    sliced = base[:, 0:1, :]                            # 形状 [4,1,16]，size-1 维 stride 任意
    print(f"  输入 is_contiguous()={sliced.is_contiguous()} stride={tuple(sliced.stride())}")
    tail_cpu = torch.arange(4 * 3 * 16, dtype=torch.float32).reshape(4, 3, 16) + 1.0
    tail = tail_cpu.to("corex")
    out = torch.empty(4, 4, 16, device="corex")
    try:
        got = torch.cat((sliced, tail), dim=1, out=out)
        ok = torch.equal(readback(got), torch.cat((base_cpu[:, 0:1, :], tail_cpu), dim=1))
        print(f"  {'OK  ' if ok else 'FAIL'} cat([4,1,16] + [4,3,16], dim=1) 数值一致={ok}")
        cat_size1_ok = ok
    except Exception as exc:                            # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        cat_size1_ok = False

    print("\n=== 结论 ===")
    missing = [key for key, value in results.items() if value is None]
    print(f"  用例数 {len(results)}，失败 {len(missing)}"
          + (f"：{missing}" if missing else ""))
    print(f"  [4,1,16] 那个 stride 回归：{'通过' if cat_size1_ok else '失败'}")
    all_ok = not missing and cat_size1_ok
    print("COPY_CAT_PERF_PASS" if all_ok else "COPY_CAT_PERF_FAIL")
    print("COPY_CAT_PERF_DONE")
    raise SystemExit(0 if all_ok else 1)


main()
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
