set -u
# 目的：验收 RMSNorm / LayerNorm（`_ops_reduce.py`）与 gather_rows（`_ops_index.py`）。
#   1. norm 与 CPU **双精度**参照比，报 max_abs/max_rel；同时打印「一遍法（E[x²]-mean²）
#      与两遍法」的方差差 —— 那是 `_layernorm_kernel` 用单遍算法的代价，量出来写进文档，
#      而不是嘴上说「误差够小」；
#   2. 两个边界：常量行（layernorm 输出必须精确为 0）、cols 非 2 的幂（1000 / 4097）；
#   3. gather_rows 是纯搬运，要求**逐字节相等**（含重复下标与乱序下标）；
#   4. 性能：norm 报 GB/s（读 x + 读 w + 写 out），gather_rows 报 GB/s（读 1 行 + 写 1 行）。
#
# 载荷：`work/ops/payload.tgz`。
# 跑法：python tools/remote.py sbx-run probes/182_norm_index.sh --name 182_norm_index --timeout 600
# 判定：NORMIDX_VERDICT PASS，退出码如实。

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

cat > "$W/norm_index_check.py" <<'PYEOF'
"""RMSNorm / LayerNorm / gather_rows：正确性 + 带宽。无 torch 进程。"""

import ctypes
import math
import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as elt
from corex_torch import _ops_reduce as rdc
from corex_torch import _ops_index as idx_


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def pack(values):
    return b"".join(struct.pack("<f", v) for v in values)


def unpack(raw, n):
    return [struct.unpack_from("<f", raw, 4 * i)[0] for i in range(n)]


def to_dev(values):
    raw = pack(values)
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def from_dev(ptr, n):
    buf = ctypes.create_string_buffer(4 * n)
    rt.memcpy(buf, ptr, 4 * n, rt.MEMCPY_DEVICE_TO_HOST)
    return unpack(buf.raw, n)


def make_matrix(rows, cols, seed=0):
    """fp32 精确可表示值：|x| ≤ 50，步长 0.5。"""
    return [float(((i * 37 + j * 13 + seed) % 199) - 99) * 0.5
            for i in range(rows) for j in range(cols)]


def make_weight(cols):
    return [1.0 + ((j % 5) - 2) * 0.25 for j in range(cols)]     # 0.5 / 0.75 / 1.0 / 1.25 / 1.5


# --------------------------------------------------------------------------- #
# §1 正确性
# --------------------------------------------------------------------------- #

def correctness():
    print("=== §1 正确性（fp32；参照 = CPU 双精度）===")
    ok = total = 0
    eps = 1e-6

    for rows, cols in ((7, 512), (5, 1000), (9, 4097)):
        x = make_matrix(rows, cols)
        w = make_weight(cols)
        px = to_dev(x)
        pw = to_dev(w)
        po = rt.malloc(rows * cols * 4)

        # --- RMSNorm ---
        ref = []
        for r in range(rows):
            row = x[r * cols:(r + 1) * cols]
            ms = sum(v * v for v in row) / cols
            rstd = 1.0 / math.sqrt(ms + eps)
            ref.extend([v * rstd * w[j] for j, v in enumerate(row)])
        rdc.rms_norm(px, pw, po, rows, cols, eps, "fp32")
        rt.synchronize()
        got = from_dev(po, rows * cols)
        ma = max(abs(g - r) for g, r in zip(got, ref))
        max_rel = max(abs(g - r) / (abs(r) if abs(r) > 1e-30 else 1.0)
                      for g, r in zip(got, ref))
        good = max_rel <= 1e-5
        total += 1
        ok += 1 if good else 0
        print(f"  rms_norm  rows={rows:2d} cols={cols:4d}  max_abs={ma:.3e} "
              f"max_rel={max_rel:.3e} → {good}")

        # --- LayerNorm（并用两种方差算法标定「单遍算法」的代价）---
        ref2 = []
        var_gap = 0.0
        for r in range(rows):
            row = x[r * cols:(r + 1) * cols]
            mean = sum(row) / cols
            var2 = sum((v - mean) ** 2 for v in row) / cols            # 两遍法
            var1 = sum(v * v for v in row) / cols - mean * mean        # 一遍法（内核用的）
            var_gap = max(var_gap, abs(var1 - var2))
            rstd = 1.0 / math.sqrt(var2 + eps)
            ref2.extend([(v - mean) * rstd * w[j] for j, v in enumerate(row)])
        rdc.layernorm(px, pw, po, rows, cols, eps, "fp32")
        rt.synchronize()
        got = from_dev(po, rows * cols)
        ma = max(abs(g - r) for g, r in zip(got, ref2))
        max_rel = max(abs(g - r) / (abs(r) if abs(r) > 1e-30 else 1.0)
                      for g, r in zip(got, ref2))
        good = max_rel <= 1e-5
        total += 1
        ok += 1 if good else 0
        print(f"  layernorm rows={rows:2d} cols={cols:4d}  max_abs={ma:.3e} "
              f"max_rel={max_rel:.3e} → {good}   （一遍法vs两遍法方差差={var_gap:.3e}）")

        # --- 常量行：layernorm 必须精确输出 0 ---
        const_val = 3.5
        xc = [const_val] * (rows * cols)
        pc = to_dev(xc)
        rdc.layernorm(pc, pw, po, rows, cols, eps, "fp32")
        rt.synchronize()
        gotc = from_dev(po, rows * cols)
        allzero = all(v == 0.0 for v in gotc)
        # rms_norm 的常量行应当是 w 本身（rstd = 1/sqrt(c^2+eps) ≈ 1/|c|）
        rdc.rms_norm(pc, pw, po, rows, cols, eps, "fp32")
        rt.synchronize()
        gotr = from_dev(po, rows * cols)
        exp_r = [const_val * (1.0 / math.sqrt(const_val * const_val + eps)) * w[j]
                 for j in range(cols)]
        ma_r = max(abs(g - e) for g, e in zip(gotr[:cols], exp_r))
        total += 1
        ok += 1 if (allzero and ma_r <= 1e-5) else 0
        print(f"  常量行边界（rows={rows} cols={cols}）  layernorm 全 0={allzero}  "
              f"rms_norm vs w 的 max_abs={ma_r:.3e}")
        for p in (px, pw, po, pc):
            rt.free(p)

    # --- gather_rows：纯搬运，要求逐字节相等 ---
    for rows, cols, table_rows in ((7, 1000, 64), (5, 4097, 65), (33, 512, 40)):
        table = make_matrix(table_rows, cols, seed=11)
        # 乱序 + 重复下标，覆盖「同一行被多次取」和「跳着取」
        idx_list = [((i * 7 + 3) % table_rows) for i in range(rows)]
        idx_list[0] = 0
        if rows > 1:
            idx_list[1] = idx_list[0]                       # 故意重复
        src = to_dev(table)
        pidx = rt.malloc(rows * 4)
        idx32 = (ctypes.c_int32 * rows)(*idx_list)
        rt.memcpy(pidx, idx32, rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
        po = rt.malloc(rows * cols * 4)
        idx_.gather_rows(src, pidx, po, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(po, rows * cols)
        ref = []
        for r in range(rows):
            s = idx_list[r]
            ref.extend(table[s * cols:(s + 1) * cols])
        exact = pack(got) == pack(ref)
        # i64 下标也要能跑
        idx64 = (ctypes.c_int64 * rows)(*idx_list)
        pidx64 = rt.malloc(rows * 8)
        rt.memcpy(pidx64, idx64, rows * 8, rt.MEMCPY_HOST_TO_DEVICE)
        po2 = rt.malloc(rows * cols * 4)
        idx_.gather_rows(src, pidx64, po2, rows, cols, "fp32", idx_dtype="i64")
        rt.synchronize()
        exact64 = pack(from_dev(po2, rows * cols)) == pack(ref)
        total += 1
        ok += 1 if (exact and exact64) else 0
        print(f"  gather_rows rows={rows:2d} cols={cols:4d} 逐字节相等(i32)={exact} (i64)={exact64}")
        for p in (src, pidx, po, pidx64, po2):
            rt.free(p)

    print(f"§1 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §2 带宽
# --------------------------------------------------------------------------- #

_FLUSH = {}


def flush_l2():
    if "ptr" not in _FLUSH:
        m = 64 << 20
        _FLUSH["ptr"] = rt.malloc(m * 4)
        _FLUSH["n"] = m
    elt.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench(fn, reps=20, warmup=3):
    pairs = [(rt.Event(), rt.Event()) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    rt.synchronize()
    for i in range(reps):
        flush_l2()
        pairs[i][0].record()
        fn()
        pairs[i][1].record()
    rt.synchronize()
    ts = sorted(a.elapsed_ms(b) / 1000.0 for a, b in pairs)
    for a, b in pairs:
        a.close()
        b.close()
    return ts[len(ts) // 2]


def bandwidth():
    print("=== §2 带宽（L2 每轮刷 256 MiB，事件中位数）===")
    rows, cols = 1024, 1000
    x = make_matrix(rows, cols)
    w = make_weight(cols)
    px, pw = to_dev(x), to_dev(w)
    po = rt.malloc(rows * cols * 4)
    body = rows * cols * 4

    for name, fn, bufs in (
        ("rms_norm", lambda: rdc.rms_norm(px, pw, po, rows, cols, 1e-6, "fp32"), 3),
        ("layernorm", lambda: rdc.layernorm(px, pw, po, rows, cols, 1e-6, "fp32"), 3),
    ):
        dt = bench(fn)
        tb = body * bufs
        print(f"  {name:12s} rows={rows} cols={cols}  {dt * 1e6:8.1f} us  "
              f"读x + 读w + 写out = {tb / 1048576:.2f} MiB → {tb / dt / 1e9:7.1f} GB/s"
              f"（w 只有 {cols * 4 / 1024:.1f} KiB，重复读靠 L1）")

    # gather_rows：读 rows 行 + 写 rows 行
    table_rows = 4096
    table = make_matrix(table_rows, cols, seed=13)
    src = to_dev(table)
    idx_list = [((i * 37 + 11) % table_rows) for i in range(rows)]
    pidx = rt.malloc(rows * 4)
    rt.memcpy(pidx, (ctypes.c_int32 * rows)(*idx_list), rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
    dt = bench(lambda: idx_.gather_rows(src, pidx, po, rows, cols, "fp32"))
    tb = 2 * body
    print(f"  {'gather_rows':12s} rows={rows} cols={cols}  {dt * 1e6:8.1f} us  "
          f"读+写 {tb / 1048576:.2f} MiB → {tb / dt / 1e9:7.1f} GB/s")
    for p in (px, pw, po, src, pidx):
        rt.free(p)
    print("  cache: reduce", rdc.cache_info()["kernels"], "个内核 / index",
          idx_.cache_info()["kernels"], "个内核已编")


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0), " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE))
    ok, total = correctness()
    bandwidth()
    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if ok == total else "FAIL"
    print(f"正确性 {ok}/{total}")
    print("NORMIDX_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. norm + gather：正确性 / 带宽 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/norm_index_check.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE182_DONE py_rc=$PYRC"
exit "$PYRC"
