set -u
# 目的：验收矩阵乘（`_ops_matmul.py` 的 `mm` / `bmm`，`tl.dot` 走 FMA，不碰矩阵单元）。
#   1. 正确性分两档，故意的：
#      a) **dyadic 输入**（元素取 ±0.5/±1/±2 这类 fp32 精确值）时，fp32 累加**本来就精确**，
#         所以要求与 CPU 参照**逐字节相等** —— 这一档抓的是布局/索引错（行主序算错一维、
#         批量偏移算错），那才是矩阵乘最容易出的错；
#      b) **非 dyadic 输入**（0.1 这种）时 fp32 累加有舍入，改为报 max_abs/max_rel，并与
#         「CPU 顺序 fp32 累加 vs 双精度」的差值对照 —— 那是累加顺序差异的自然量级。
#   2. 尾块/非整除：M/N/K 都取非分块整数倍（17×19×23、37×53×29）。
#   3. 性能：M=N=K=256 与 512 的 GFLOP/s，并与两个参照并列：
#        Triton FMA 基线 105 GFLOP/s（corex-triton §14.3）、厂商 TCU 1577 GFLOP/s（只当天花板）。
#      同时扫一遍 (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)，用来定默认值。
#
# 载荷：`work/ops/payload.tgz`。
# 跑法：python tools/remote.py sbx-run probes/181_mm.sh --name 181_mm --timeout 600
# 判定：MM_VERDICT PASS，退出码如实。

PY=$HOME/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
mkdir -p "$W"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

echo "=== 1. 解包载荷 ==="
if [ ! -f "$W/payload.tgz" ]; then echo "PAYLOAD_MISSING $W/payload.tgz"; exit 66; fi
rm -rf "$W/corex-torch" "$W/corex-triton"
tar xzf "$W/payload.tgz" -C "$W" || exit 67
export PYTHONPATH=$W/corex-torch/src
export COREX_TRITON_SRC=$W/corex-triton/src

cat > "$W/mm_check.py" <<'PYEOF'
"""矩阵乘：dyadic 逐字节 + 非 dyadic 误差 + GFLOP/s + 分块扫描。无 torch 进程。"""

import ctypes
import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as elt
from corex_torch import _ops_matmul as mm_

FMT = "<f"


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def pack(values):
    return b"".join(struct.pack(FMT, v) for v in values)


def unpack(raw, n):
    return [struct.unpack_from(FMT, raw, 4 * i)[0] for i in range(n)]


def round_f32(v):
    return struct.unpack(FMT, struct.pack(FMT, v))[0]


def to_dev(values):
    raw = pack(values)
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def from_dev(ptr, n):
    buf = ctypes.create_string_buffer(4 * n)
    rt.memcpy(buf, ptr, 4 * n, rt.MEMCPY_DEVICE_TO_HOST)
    return unpack(buf.raw, n)


def dyadic_matrix(rows, cols, seed=1):
    """值取 ±0.5/±1/±2 这类 fp32 精确值：K 段累加在没有大指数跨度时是精确的。"""
    table = (0.5, -1.0, 2.0, -0.5, 1.0, -2.0, 0.0)
    return [table[(i * 7 + j * 3 + seed) % len(table)] for i in range(rows) for j in range(cols)]


def real_matrix(rows, cols, seed=1):
    """非 dyadic：0.1 的倍数，让 fp32 累加真的产生舍入。"""
    return [float(((i * 11 + j * 5 + seed) % 17) - 8) * 0.1 - 0.1
            for i in range(rows) for j in range(cols)]


def cpu_mm(a, b, M, N, K):
    """CPU 双精度参照（O(MNK)，只用于小尺寸）。"""
    c = [0.0] * (M * N)
    for i in range(M):
        ib = i * K
        ic = i * N
        for k in range(K):
            aik = a[ib + k]
            if aik == 0.0:
                continue
            kb = k * N
            for j in range(N):
                c[ic + j] += aik * b[kb + j]
    return c


def cpu_mm_f32_sequential(a, b, M, N, K):
    """CPU 侧「一步步 fp32 累加」的参照：用来标定累加顺序误差的量级。"""
    c = [0.0] * (M * N)
    for i in range(M):
        ib = i * K
        ic = i * N
        for k in range(K):
            aik = a[ib + k]
            kb = k * N
            for j in range(N):
                c[ic + j] = round_f32(c[ic + j] + aik * b[kb + j])
    return c


def cpu_mm_element(a, b, N, K, i, j):
    """只算 (i, j) 一个元素的双精度值（大尺寸抽查用）。"""
    s = 0.0
    ib = i * K
    for k in range(K):
        s += a[ib + k] * b[k * N + j]
    return s


# --------------------------------------------------------------------------- #
# §1 正确性
# --------------------------------------------------------------------------- #

def correctness():
    print("=== §1 正确性 ===")
    ok = total = 0

    print("--- a) dyadic 输入：要求与 CPU 参照逐字节相等（抓布局/索引错）---")
    for M, N, K in ((8, 8, 8), (17, 19, 23), (37, 53, 29), (64, 64, 64), (100, 33, 65)):
        a = dyadic_matrix(M, K, seed=1)
        b = dyadic_matrix(K, N, seed=2)
        pa, pb = to_dev(a), to_dev(b)
        pc = rt.malloc(M * N * 4)
        mm_.mm(pa, pb, pc, M, N, K, "fp32")
        rt.synchronize()
        got = from_dev(pc, M * N)
        ref = cpu_mm(a, b, M, N, K)
        exact = pack(got) == pack(ref)
        ma = max(abs(g - r) for g, r in zip(got, ref))
        total += 1
        ok += 1 if exact else 0
        print(f"  mm {M:4d}x{N:4d}x{K:4d}  逐字节相等={exact} max_abs={ma:.3e}")
        for p in (pa, pb, pc):
            rt.free(p)

    print("--- b) 非 dyadic 输入：报误差，并用「顺序 fp32 累加」标定量级 ---")
    for M, N, K in ((32, 32, 32), (17, 19, 23), (37, 53, 29)):
        a = real_matrix(M, K, seed=3)
        b = real_matrix(K, N, seed=4)
        pa, pb = to_dev(a), to_dev(b)
        pc = rt.malloc(M * N * 4)
        mm_.mm(pa, pb, pc, M, N, K, "fp32")
        rt.synchronize()
        got = from_dev(pc, M * N)
        dbl = cpu_mm(a, b, M, N, K)
        seq = cpu_mm_f32_sequential(a, b, M, N, K)
        gap = max(abs(s - d) for s, d in zip(seq, dbl))
        ma = max(abs(g - d) for g, d in zip(got, dbl))
        # 相对误差只在「参照本身不小」的元素上算：矩阵乘会出现大量相消到 ~0 的元素，
        # 拿它们做分母会让 max_rel 变成 1e8 这种无意义的值（分母趋 0）。
        worst = 0.0
        for g, d in zip(got, dbl):
            if abs(d) > 1e-3:
                worst = max(worst, abs(g - d) / abs(d))
        limit = 8.0 * gap + 1e-6 * max(1.0, max(abs(x) for x in dbl))
        total += 1
        ok += 1 if ma <= limit else 0
        print(f"  mm {M:4d}x{N:4d}x{K:4d}  max_abs={ma:.3e} max_rel(|ref|>1e-3)={worst:.3e} "
              f"限={limit:.3e}（顺序fp32误差={gap:.3e}）→ {ma <= limit}")
        for p in (pa, pb, pc):
            rt.free(p)

    print("--- c) bmm（批量）：dyadic，逐字节 ---")
    for B, M, N, K in ((2, 8, 8, 8), (3, 13, 17, 11), (4, 32, 16, 24)):
        a = dyadic_matrix(B * M, K, seed=5)
        b = dyadic_matrix(B * K, N, seed=6)
        pa, pb = to_dev(a), to_dev(b)
        pc = rt.malloc(B * M * N * 4)
        mm_.bmm(pa, pb, pc, B, M, N, K, "fp32")
        rt.synchronize()
        got = from_dev(pc, B * M * N)
        ref = []
        for bi in range(B):
            ab = a[bi * M * K:(bi + 1) * M * K]
            bb = b[bi * K * N:(bi + 1) * K * N]
            ref.extend(cpu_mm(ab, bb, M, N, K))
        exact = pack(got) == pack(ref)
        total += 1
        ok += 1 if exact else 0
        print(f"  bmm B={B:2d} {M:3d}x{N:3d}x{K:3d}  逐字节相等={exact}")
        for p in (pa, pb, pc):
            rt.free(p)

    print(f"§1 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §2 性能
# --------------------------------------------------------------------------- #

_FLUSH = {}


def flush_l2():
    if "ptr" not in _FLUSH:
        m = 64 << 20
        _FLUSH["ptr"] = rt.malloc(m * 4)
        _FLUSH["n"] = m
    elt.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench(fn, reps, warmup=2):
    pairs = [(rt.Event(), rt.Event()) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    rt.synchronize()
    for i in range(reps):
        flush_l2()                       # 与 corex-triton §15.2 的 do_bench 口径一致
        pairs[i][0].record()
        fn()
        pairs[i][1].record()
    rt.synchronize()
    ts = sorted(a.elapsed_ms(b) / 1000.0 for a, b in pairs)
    for a, b in pairs:
        a.close()
        b.close()
    return ts[len(ts) // 2]


def perf():
    print("=== §2 性能（GFLOP/s，每轮刷 256 MiB L2，事件中位数）===")
    out = {}
    for dim in (256, 512):
        M = N = K = dim
        a = real_matrix(M, K, seed=7)
        b = real_matrix(K, N, seed=8)
        pa, pb = to_dev(a), to_dev(b)
        pc = rt.malloc(M * N * 4)
        mm_.mm(pa, pb, pc, M, N, K, "fp32")
        rt.synchronize()
        got = from_dev(pc, M * N)
        # 大尺寸只抽查几个元素（全量 CPU 参照是 O(MNK)，512³ 在 Python 里不现实）
        spots = [(0, 0), (M // 2, N // 2), (M - 1, N - 1), (1, N - 3)]
        diffs = []
        for i, j in spots:
            r = cpu_mm_element(a, b, N, K, i, j)
            if abs(r) > 1e-3:            # 相消到 0 的元素不能做相对误差
                diffs.append(abs(got[i * N + j] - r) / abs(r))
        reps = 20 if dim <= 256 else 10
        dt = bench(lambda: mm_.mm(pa, pb, pc, M, N, K, "fp32"), reps=reps)
        gf = mm_.flops(M, N, K) / dt / 1e9
        out[dim] = gf
        print(f"  mm {dim}^3   {dt * 1e6:9.1f} us  {gf:7.1f} GFLOP/s   "
              f"抽查 max_rel(|ref|>1e-3)={max(diffs):.3e}")
        for p in (pa, pb, pc):
            rt.free(p)

    # bmm 的吞吐（用 256 的同一组数据，B=8）
    B, M, N, K = 8, 256, 256, 256
    a = real_matrix(B * M, K, seed=9)
    b = real_matrix(B * K, N, seed=10)
    pa, pb = to_dev(a), to_dev(b)
    pc = rt.malloc(B * M * N * 4)
    dt = bench(lambda: mm_.bmm(pa, pb, pc, B, M, N, K, "fp32"), reps=10)
    gf = mm_.flops(M, N, K, batch=B) / dt / 1e9
    print(f"  bmm B={B} M=N=K={M}  {dt * 1e6:9.1f} us  {gf:7.1f} GFLOP/s")
    out["bmm8_256"] = gf
    for p in (pa, pb, pc):
        rt.free(p)

    print("=== §3 分块扫描（mm 256^3，定默认值用）===")
    M = N = K = 256
    a = real_matrix(M, K, seed=7)
    b = real_matrix(K, N, seed=8)
    pa, pb = to_dev(a), to_dev(b)
    pc = rt.malloc(M * N * 4)
    best = (None, -1.0)
    configs = [
        (16, 16, 16, 1, 1), (16, 16, 32, 1, 1), (16, 16, 32, 1, 3),
        (16, 16, 64, 2, 1), (32, 32, 32, 1, 1), (32, 32, 32, 4, 1),
        (32, 32, 32, 1, 3), (64, 64, 16, 4, 1), (32, 64, 32, 4, 1),
        (16, 32, 32, 2, 1),
    ]
    for bm, bn, bk, warps, stages in configs:
        try:
            fn = (lambda bm=bm, bn=bn, bk=bk, w=warps, s=stages:
                  mm_.mm(pa, pb, pc, M, N, K, "fp32", block_m=bm, block_n=bn, block_k=bk,
                         num_warps=w, num_stages=s))
            fn()
            rt.synchronize()
            dt = bench(fn, reps=10)
            gf = mm_.flops(M, N, K) / dt / 1e9
            flag = ""
            if gf > best[1]:
                best = ((bm, bn, bk, warps, stages), gf)
                flag = " ← 目前最好"
            print(f"  BM={bm:3d} BN={bn:3d} BK={bk:3d} warps={warps} stages={stages}  "
                  f"{dt * 1e6:9.1f} us  {gf:7.1f} GFLOP/s{flag}")
        except Exception as exc:                        # noqa: BLE001 - 配置编不过也是信息
            print(f"  BM={bm:3d} BN={bn:3d} BK={bk:3d} warps={warps} stages={stages}  "
                  f"失败 {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
    for p in (pa, pb, pc):
        rt.free(p)

    print("=== §4 参照 ===")
    print("  Triton FMA 基线（corex-triton §14.3，256^3）      105.0 GFLOP/s")
    print("  厂商 TCU GEMM（docs/00-FACTS.md §9，矩阵单元）   1577.0 GFLOP/s（只当天花板）")
    print(f"  我们 mm 256^3                                    {out[256]:7.1f} GFLOP/s "
          f"→ 基线 {out[256] / 105.0:.2f}x / 天花板 {out[256] / 1577.0:.3f}x")
    print(f"  我们 mm 512^3                                    {out[512]:7.1f} GFLOP/s")
    print(f"  扫描最好配置 {best[0]}  {best[1]:.1f} GFLOP/s")
    print("  默认值 =", (mm_.DEFAULT_BLOCK_M, mm_.DEFAULT_BLOCK_N, mm_.DEFAULT_BLOCK_K,
                        mm_.DEFAULT_NUM_WARPS, mm_.DEFAULT_NUM_STAGES))
    return out


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0), " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE))
    print("默认分块 =", (mm_.DEFAULT_BLOCK_M, mm_.DEFAULT_BLOCK_N, mm_.DEFAULT_BLOCK_K),
          " warps =", mm_.DEFAULT_NUM_WARPS, " stages =", mm_.DEFAULT_NUM_STAGES)
    ok, total = correctness()
    perf()
    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if ok == total else "FAIL"
    print(f"正确性 {ok}/{total}")
    print("MM_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 矩阵乘：正确性 + GFLOP/s + 分块扫描 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/mm_check.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE181_DONE py_rc=$PYRC"
exit "$PYRC"
