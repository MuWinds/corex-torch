set -u
# 目的：验收 corex-torch 自己的 Triton 适配层（src/corex_torch/_triton.py）——不是再证明一遍
# 「Triton 能在卡上跑」，那件事 corex-triton 已经做过。这里要证的是**我们这层写得对**：
#   1. 整条路（编译 → 装载 → 启动）在没有 torch 的进程里通；
#   2. 编译复用 corex-triton 的 CorexBackend，运行期用我们自己的 libcuda/libcudart 绑定；
#   3. 结果与 CPU 参照逐字节相等（用 fp32 精确可表示值，不用「误差够小」这种口径）。
# 之所以要无 torch：设备层跑在上游 CPU 版 torch 2.8 上，没有 torch.cuda。这条路上任何
# 一处偷偷 import 厂商 torch 2.1 都会让 torch 2.8 环境里直接崩。
#
# 为什么要落成文件再跑：@triton.jit 要用 inspect.getsourcelines 读内核源码，
# 从 stdin（heredoc）进来的函数没有源码可读，会抛 OSError: could not get source code。
PY=$HOME/venvs/triton34/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export COREX_TRITON_SRC=$W/corex-triton-src
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"
echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
"$PY" -c 'import triton;print("TRITON", triton.__version__)' 2>&1 | tail -2

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/adapter_check.py" <<'PYEOF'
import ctypes
import sys
import time

sys.path.insert(0, sys.argv[1])

def torch_modules():
    return sorted(k for k in sys.modules if k == "torch" or k.startswith("torch."))

import corex_torch
from corex_torch import _runtime as rt, _triton as ct

print("[start]                torch_modules =", len(torch_modules()), torch_modules()[:3])
driver = ct.activate()
print("[after_activate]       torch_modules =", len(torch_modules()), torch_modules()[:3])

import triton
import triton.language as tl

@triton.jit
def vecadd(a, b, c, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(c + off, tl.load(a + off, mask=mask) + tl.load(b + off, mask=mask), mask=mask)

SIG = {"a": "*fp32", "b": "*fp32", "c": "*fp32", "n": "i32", "BLOCK": "constexpr"}

def to_dev(values):
    buf = (ctypes.c_float * len(values))(*values)
    nbytes = ctypes.sizeof(buf)
    ptr = rt.malloc(nbytes)
    rt.memcpy(ptr, buf, nbytes, rt.MEMCPY_HOST_TO_DEVICE)
    return ptr, nbytes

t0 = time.perf_counter()
kernel = ct.compile_kernel(vecadd, signature=SIG, constexprs={"BLOCK": 128})
t1 = time.perf_counter()
print(f"[compiled]             {1000 * (t1 - t0):.0f} ms  asm_keys={sorted(kernel.asm.keys())}")
print("[after_compile]        torch_modules =", len(torch_modules()), torch_modules()[:3])
print("                       cubin_bytes =", len(kernel.asm["cubin"]),
      " name =", repr(getattr(kernel.metadata, "name", None)))

def run(n):
    host_a = [i * 0.5 for i in range(n)]
    host_b = [i * 0.25 for i in range(n)]
    expect = [x + y for x, y in zip(host_a, host_b)]
    pa, nbytes = to_dev(host_a)
    pb, _ = to_dev(host_b)
    pc = rt.malloc(nbytes)
    grid = (triton.cdiv(n, 128),)
    t = time.perf_counter()
    # 故意只给一维网格：launch() 要负责补齐成 3 元组（CompiledKernel 的 runner 自己不补）。
    ct.launch(kernel, grid, pa, pb, pc, n)
    rt.synchronize()
    dt = 1000 * (time.perf_counter() - t)
    out = (ctypes.c_float * n)()
    rt.memcpy(out, pc, nbytes, rt.MEMCPY_DEVICE_TO_HOST)
    got = list(out)
    for dev in (pa, pb, pc):
        rt.free(dev)
    max_diff = max(abs(g - e) for g, e in zip(got, expect))
    print(f"[launched] n={n:5d} grid={grid} {dt:6.2f} ms  equal={got == expect} "
          f"max_abs_diff={max_diff:.3e}  got[0:3]={got[:3]}")
    return got == expect

ok_main = run(4096)
ok_tail = run(4000)          # 尾块：4000 不是 128 的整数倍，越界部分必须被掩码挡住
print("[end]                  torch_modules =", len(torch_modules()), torch_modules()[:3])
print("ADAPTER_VERDICT", "PASS" if (ok_main and ok_tail) else "FAIL")
PYEOF

echo "=== 1. 编译 + 启动（corex_torch._triton，全程无 torch）==="
timeout --foreground -k 5 240 "$PY" -B -u "$W/adapter_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
echo "=== 2. 卡还在吗 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
# 退出码要如实反映 python 段的结果：脚本最后一句若是 echo，runner 里永远是 rc=0，
# 失败就得靠人翻日志——2026-11-07 踩过一次（004 那版 rc=0 但日志里全是 traceback）。
exit $PYRC
