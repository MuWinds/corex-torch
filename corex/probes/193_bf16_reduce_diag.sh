set -u
# 目的：给 `_bf16.rewrite_bf16_conversions` 定位漏改的那一类转换。
#
# 现象（probes/192）：元素级 bf16 **33/33 逐字节全对**（说明改写层对那批内核生效了），
# 但规约类（sum_last / softmax / rms_norm / layernorm 与 mm）仍报
#     LLVM ERROR: Cannot select: f32 = fp_extend … _ops_reduce.py:104:72
# 本探针把失败内核的 biir 落盘，把里面 `fpext` / `fptrunc` 的**原始行**打出来，并对照
# `_bf16.rewrite_bf16_conversions` 的命中数——用来区分「正则没匹配上」还是「匹配上了但改写不对」。
#
# 跑法：python tools/remote.py sbx-run probes/193_bf16_reduce_diag.sh --name 193_bf16_diag --timeout 600

PY=$HOME/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
mkdir -p "$W"

echo "=== 0. 验卡 ==="
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
# 让 corex-triton 在进 llc 之前把 biir 落盘（失败也留得下来）
export COREX_DUMP_IR=$W/diag_biir.ll

cat > "$W/bf16_diag.py" <<'PYEOF'
"""打印失败内核的 fpext/fptrunc 原始行，并检查 _bf16 改写器的命中数。"""

import glob
import os
import re
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _bf16
import corex_torch._ops_reduce as rdc

W = os.environ.get("TB_W", ".")


def main():
    print("device =", rt.device_name(0))
    print("_bf16 模块:", _bf16.__file__)

    # 直接把「展宽」那一类 IR 交给改写器，看它认不认（这是本探针的核心判断）
    samples = [
        # 标量形式（元素级内核里出现的形态，已知能改）
        "  %5 = fpext bfloat %4 to float\n",
        # 向量形式（规约内核里出现的形态，怀疑改不到）
        "  %9 = fpext <8 x bfloat> %8 to <8 x float>\n",
        "  %10 = fptrunc <8 x float> %9 to <8 x bfloat>\n",
        "  %11 = fptrunc float %5 to bfloat\n",
    ]
    print("=== §1 改写器对两种形式的识别能力 ===")
    for s in samples:
        out, hits = _bf16.rewrite_bf16_conversions(s)
        changed = out != s
        print(f"  命中={hits} 改动={changed}  输入: {s.strip()}")
        if changed:
            print(f"                          输出: {out.strip()}")

    print("=== §2 真机编译 bf16 规约内核，落盘 biir 里 fpext/fptrunc 的原文 ===")
    try:
        rdc.sum_last(0, 0, 0, 0, "bf16")     # 只为触发编译；n=0 会直接返回，换成下面的真调用
    except Exception:
        pass
    n = 256
    src = rt.malloc(n * 2)
    out = rt.malloc(2)
    err = None
    try:
        rdc.sum_last(src, out, 1, n, "bf16")
        rt.synchronize()
        print("  ⚠ 竟然编译成功了——那问题不在改写层，需要另找")
    except Exception as exc:                            # noqa: BLE001
        err = exc
        print(f"  编译失败（预期）: {type(exc).__name__}: "
              f"{str(exc).replace(chr(10), ' ')[:200]}")
    rt.free(src)
    rt.free(out)

    hits = sorted(glob.glob(os.path.join(W, "diag_biir*.ll")), key=os.path.getmtime)
    if not hits:
        print("  （没有 biir dump）")
        return 0
    path = hits[-1]
    text = open(path, encoding="utf-8", errors="replace").read()
    print(f"  dump={os.path.basename(path)} 共 {len(text.splitlines())} 行")

    print("=== §3 dump 里所有 bfloat 相关的转换行 ===")
    shown = 0
    for i, ln in enumerate(text.splitlines(), 1):
        if ("fpext" in ln or "fptrunc" in ln or "bfloat2float" in ln
                or "float2bfloat" in ln):
            print(f"  {i:5d}: {ln.strip()[:180]}")
            shown += 1
            if shown >= 12:
                print("  …（只列前 12 行）")
                break
    if shown == 0:
        print("  （没有 fpext/fptrunc 行——那 llc 的 fp_extend 是别处来的）")

    # 把整份 biir 交给改写器，看它总共能改多少处
    _, total_hits = _bf16.rewrite_bf16_conversions(text)
    raw_fpext = text.count("fpext")
    raw_fptrunc = text.count("fptrunc")
    print(f"=== §4 计数：dump 里 fpext={raw_fpext} fptrunc={raw_fptrunc}；"
          f"改写器整份处理命中={total_hits} ===")
    print(f"  结论: " + ("改写器漏掉了向量形式（命中数 < fpext+fptrunc 的原始计数）"
                        if total_hits < raw_fpext + raw_fptrunc else
                        "改写器全覆盖，问题不在这里"))
    print("DIAG_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 诊断 ==="
timeout --foreground -k 5 240 "$PY" -B -u "$W/bf16_diag.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE193_DONE py_rc=$PYRC"
exit "$PYRC"
