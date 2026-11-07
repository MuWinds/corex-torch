set -u
# 目的：查 `bmm [4,1,16] @ [4,16,6]` 单次 **2505 µs**（同量级的 `[4,1,6] @ [4,6,16]` 只有 85.6 µs）。
#
# 口径（按 Lead 要求）：同一形状跑 **200 次**、**循环外同步一次**、取总时间/次数；另给中位数。
# 三层拆分：
#   L1  `_ops_matmul.bmm(...)`        —— 我的 Python 包装层（dict 查缓存 + launch 参数打包）
#   L2  直接 `_triton.launch(kernel)` —— 绕过包装层，只剩 launch
#   L3  形状扫描                      —— 找出到底是 B / M / N / K 哪个维度触发的
# 另外两条辅助测量，用来排掉「不是内核」的可能：
#   A. 分配器：`rt.malloc(n)+free` 各种尺寸各 200 次（输出张量尺寸差一个数量级）
#   B. 编译缓存：调用前后 `_KERNEL_CACHE` 的长度与内核对象 identity（排除每次重编）
#
# 判据：L1/L2 在两形状上应当同量级（都是几次 launch 的量级）。若确实同量级，那 2.5 ms 就在
# 我的层之上（aten 分发 / `_empty` / 探针 lambda 里的张量构造），我把证据和下一步测量给 Lead。
#
# 跑法：python tools/remote.py sbx-run probes/198_bmm_repro.sh --name 198_bmm --timeout 600

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
export TB_W=$W

cat > "$W/bmm_repro.py" <<'PYEOF'
"""bmm 形状计时：L1 包装层 / L2 裸 launch / 形状扫描 / 分配器 / 编译缓存。"""

import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _triton as ct
from corex_torch import _ops_matmul as mm_


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def dev_f32(n):
    ptr = rt.malloc(max(n * 4, 4))
    rt.memset(ptr, 0, n * 4) if hasattr(rt, "memset") else None
    return ptr


def fill_const(ptr, n, value):
    """用元素级 fill 把缓冲区写成常量，避免读未初始化内存（allocator 可能给脏页）。"""
    from corex_torch import _ops_elementwise as elt
    if n:
        elt.fill(ptr, n, "fp32", value)


def time_loop(fn, iters=200, warmup=10):
    """**循环外同步一次**的总时间/次数，并给出分布。

    为什么要给分布：第一次跑出「慢的那条 36.77 µs vs 中位 13.60 µs」，均值是中位的 2.7 倍。
    这是**极少数慢迭代污染均值**的典型症状，而 Lead 那个 2505 µs 很可能就是同一回事
    （50 次里混进一次 ~125 ms 的停顿 → 均值 2.5 ms）。所以后面每个结论都按中位/p99 判。
    """
    for _ in range(warmup):
        fn()
    rt.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    rt.synchronize()
    mean_us = (time.perf_counter() - t0) / iters * 1e6

    pairs = [(rt.Event(), rt.Event()) for _ in range(iters)]
    for i in range(iters):
        pairs[i][0].record()
        fn()
        pairs[i][1].record()
    rt.synchronize()
    ts = [a.elapsed_ms(b) * 1000.0 for a, b in pairs]
    for a, b in pairs:
        a.close()
    s = sorted(ts)
    stats = {
        "mean": mean_us,
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
        "p99": s[int(len(s) * 0.99)],
        "max": s[-1],
        "min": s[0],
        "slow100": sum(1 for t in ts if t > 100.0),
        "slow1000": sum(1 for t in ts if t > 1000.0),
        "top": sorted(range(len(ts)), key=lambda i: -ts[i])[:5],
        "ts": ts,
    }
    return stats


def show(label, st, extra=""):
    print(f"  {label:34s} 均值={st['mean']:8.2f} p50={st['p50']:7.2f} p90={st['p90']:8.2f} "
          f"p99={st['p99']:9.2f} max={st['max']:10.2f} us"
          f"  >100us:{st['slow100']:3d} >1ms:{st['slow1000']:2d} {extra}")


SHAPES = [
    (4, 1, 16, 6, "慢的那条 [4,1,16]@[4,16,6]"),
    (4, 1, 6, 16, "快的那条 [4,1,6]@[4,6,16]"),
    (4, 1, 16, 16, "同 B/M/K=16，换 N=16"),
    (4, 1, 6, 6, "同 B/M，K=N=6"),
    (4, 1, 1, 16, "K=1"),
    (4, 1, 16, 1, "N=1"),
    (1, 1, 16, 6, "B=1"),
    (4, 2, 16, 6, "M=2"),
    (4, 16, 16, 16, "全部 16"),
    (8, 1, 16, 6, "B=8"),
]


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    print("默认分块：block_m=%d block_n=%d block_k=%d warps=%d stages=%d" % (
        mm_.DEFAULT_BLOCK_M, mm_.DEFAULT_BLOCK_N, mm_.DEFAULT_BLOCK_K,
        mm_.DEFAULT_NUM_WARPS, mm_.DEFAULT_NUM_STAGES))

    print("=== §1 L1：`_ops_matmul.bmm` 包装层（200 次，循环外同步） ===")
    for (b, m, k, n, note) in SHAPES:
        pa = rt.malloc(max(b * m * k * 4, 4))
        pb = rt.malloc(max(b * k * n * 4, 4))
        pc = rt.malloc(max(b * m * n * 4, 4))
        fill_const(pa, b * m * k, 0.5)
        fill_const(pb, b * k * n, 0.25)
        ncache_before = len(mm_._KERNEL_CACHE)

        st = time_loop(lambda: mm_.bmm(pa, pb, pc, b, m, n, k, "fp32"))
        ncache_after = len(mm_._KERNEL_CACHE)
        show(f"B={b} M={m} K={k} N={n}",
             st, f"缓存 {ncache_before}→{ncache_after}  {note}")
        rt.free(pa)
        rt.free(pb)
        rt.free(pc)

    print("=== §2 L2：裸 launch（绕过包装层，200 次） ===")
    for (b, m, k, n, note) in SHAPES[:2]:
        pa = rt.malloc(max(b * m * k * 4, 4))
        pb = rt.malloc(max(b * k * n * 4, 4))
        pc = rt.malloc(max(b * m * n * 4, 4))
        fill_const(pa, b * m * k, 0.5)
        fill_const(pb, b * k * n, 0.25)
        mm_.bmm(pa, pb, pc, b, m, n, k, "fp32")          # 先编好，保证缓存命中
        rt.synchronize()
        kernel = list(mm_._KERNEL_CACHE.values())[-1]
        bm, bn = mm_.DEFAULT_BLOCK_M, mm_.DEFAULT_BLOCK_N
        grid = ((m + bm - 1) // bm, (n + bn - 1) // bn, b)
        # 注意：BATCH 是 constexpr，运行期参数只有 (a,b,c,M,N,K) 六个
        st = time_loop(lambda: ct.launch(kernel, grid, pa, pb, pc, m, n, k, stream=None))
        show(f"B={b} M={m} K={k} N={n} 裸launch", st, f"grid={grid}  {note}")
        rt.free(pa)
        rt.free(pb)
        rt.free(pc)

    print("=== §3 分配器：rt.malloc+free 各 200 次（输出张量只差几十字节，不该有 29 倍差） ===")
    for nbytes in (24, 64, 96, 256, 384, 4096, 65536):
        def one(nb=nbytes):
            p = rt.malloc(nb)
            rt.free(p)
        st = time_loop(one, iters=200)
        show(f"malloc({nbytes} B)", st)

    print("=== §4 首调 vs 稳态（排除「每次都重编」） ===")
    b, m, k, n = 4, 1, 16, 6
    pa = rt.malloc(b * m * k * 4)
    pb = rt.malloc(b * k * n * 4)
    pc = rt.malloc(b * m * n * 4)
    fill_const(pa, b * m * k, 0.5)
    fill_const(pb, b * k * n, 0.25)
    mm_.bmm(pa, pb, pc, b, m, n, k, "fp32")              # 首调（含编译）
    rt.synchronize()
    k1 = list(mm_._KERNEL_CACHE.values())[-1]
    t0 = time.perf_counter()
    mm_.bmm(pa, pb, pc, b, m, n, k, "fp32")
    rt.synchronize()
    one_shot = (time.perf_counter() - t0) * 1e6
    k2 = list(mm_._KERNEL_CACHE.values())[-1]
    print(f"  稳态单次（已含一次同步）= {one_shot:.2f} us；内核对象 identity 保持="
          f"{k1 is k2}；缓存 {len(mm_._KERNEL_CACHE)} 条")
    rt.free(pa)
    rt.free(pb)
    rt.free(pc)

    print("=== §5 慢迭代的分布（2000 次，看「均值被污染」这个假设成不成立） ===")
    b, m, k, n = 4, 1, 16, 6
    pa = rt.malloc(b * m * k * 4)
    pb = rt.malloc(b * k * n * 4)
    pc = rt.malloc(b * m * n * 4)
    fill_const(pa, b * m * k, 0.5)
    fill_const(pb, b * k * n, 0.25)
    st = time_loop(lambda: mm_.bmm(pa, pb, pc, b, m, n, k, "fp32"), iters=2000, warmup=20)
    show("慢的那条 x2000", st)
    slow = [(i, st["ts"][i]) for i in st["top"]]
    print(f"  最慢的 5 次（迭代序号, µs）: " +
          ", ".join(f"#{i}={t:.0f}" for i, t in slow))
    print(f"  → 若均值({st['mean']:.1f}) 远大于 p50({st['p50']:.1f})，说明均值被这几次"
          f"罕见停顿污染；decode 稳态应当按 p50 看。")
    rt.free(pa)
    rt.free(pb)
    rt.free(pc)

    print("=== §6 对照组：那 4% 的 ~500 µs 停顿是谁的？ ===")
    # 同一套计时方法，分别量「纯 Python 空循环」「只同步」「一个极小的 fill 内核」
    # 与 bmm。如果空循环也有同样的停顿率，那停顿就不是我们的 launch 造成的（宿主调度/邻居）。
    st = time_loop(lambda: None, iters=2000, warmup=20)
    show("纯 Python 空循环 x2000", st)
    st = time_loop(lambda: rt.synchronize(), iters=2000, warmup=20)
    show("只 rt.synchronize() x2000", st)
    from corex_torch import _ops_elementwise as elt
    psmall = rt.malloc(4 * 64)
    fill_const(psmall, 64, 1.0)
    st = time_loop(lambda: elt.fill(psmall, 64, "fp32", 1.0), iters=2000, warmup=20)
    show("fill(64 元素) x2000", st)
    rt.free(psmall)
    print("  → 空循环/同步里也出现同样量级、同样比例的 >100us 迭代 ⇒ 停顿是宿主环境的，"
          "不是内核或 launch 的。")

    print("[end] torch_modules =", len(torchmods()))
    print("BMMREPRO_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. bmm 形状计时 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/bmm_repro.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE198_DONE py_rc=$PYRC"
exit "$PYRC"
