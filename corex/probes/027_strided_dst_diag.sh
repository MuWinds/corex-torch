set -u
# 目的：`probes/008` 的非连续 strides 用例仍在失败，但报错信息换了两轮
# （"暂不支持非连续目的" → "copy_ 只接 corex 张量，收到 cpu"），说明我的分支把问题转移到了
# `_aten_extra._copy_impl` 里。这份探针只做一件事：把那个用例单独拿出来跑，**打印完整 traceback**，
# 看清到底是哪一层拿 CPU 张量当 corex 用。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/strided_dst.py" <<'PYEOF'
import sys
import traceback

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch
from corex_torch import _corex_shim as shim

corex_torch.register()
torch.manual_seed(9)

src_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print("src_cpu:", src_cpu.tolist(), "strides:", src_cpu.stride())

# 与 probes/008 完全一致的造法：一块 6 元素的设备张量 + 逻辑转置 stride (1,2)。
backing = torch.ones(6, device="corex")
code = corex_torch._device._scalar_type(torch.float32)
t = shim.from_blob(backing.data_ptr(), [2, 3], [1, 2], code, 0)
print("t: shape", tuple(t.shape), "stride", t.stride(), "contiguous", t.is_contiguous())

print("=== A. 非连续 corex -> CPU（t.cpu()）===")
try:
    print("  OK ->", t.cpu().tolist())
except Exception:                                       # noqa: BLE001
    print("  FAIL"); traceback.print_exc()

print("=== B. t.contiguous() ===")
try:
    c = t.contiguous()
    print("  OK ->", c.cpu().tolist(), "stride", c.stride())
except Exception:                                       # noqa: BLE001
    print("  FAIL"); traceback.print_exc()

print("=== C. aten._copy_from(连续设备张量, 非连续设备张量) ===")
try:
    dst = torch.empty(2, 3, device="corex")
    torch.ops.aten._copy_from(t, dst)
    print("  OK ->", dst.cpu().tolist())
except Exception:                                       # noqa: BLE001
    print("  FAIL"); traceback.print_exc()

print("=== D. 非连续 corex <- CPU 连续 ===")
try:
    dst2 = torch.empty(6, device="corex")
    view = dst2.as_strided([2, 3], [1, 2])
    dst2.fill_(0)
    view.copy_(src_cpu)
    print("  OK ->", dst2.cpu().tolist())
except Exception:                                       # noqa: BLE001
    print("  FAIL"); traceback.print_exc()
print("STRIDED_DST_DONE")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/strided_dst.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
