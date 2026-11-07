set -u
# 目的：设备注册端到端验收（用上游 CPU 版 torch 2.8 + 我们的 C++ shim + 算子层）。
# 这一步过了就意味着「torch 2.8 上有了一张能用的卡」：分配、H2D/D2H、逐元素算子、
# 以及 .item() 这类回主机的小动作全部打通。
#
# 为什么每一步都单独 try：PrivateUse1 的缺口是一个算子一个算子暴露的（`aten::empty` 通了
# 才会撞上 `aten::zero_`，再撞上 `_copy_from`），一次崩掉就看不到后面还缺什么。
# 这里把每个失败都记成一行，最后汇总成「还缺哪些算子」——这份清单就是下一步的待办。
#
# 前置：C++ shim 必须在登录节点编好并随 src 一起推上来（src/corex_torch/_corex_shim.so）。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
# 算子层与设备层都要能找到 corex-triton 的源码（它没打包安装，只能按路径引用）。
# 忘了这条时的症状是 `No module named 'triton_corex'`，且只在第一次要编内核时才暴露。
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"
"$PY" -c 'import torch;print("TORCH", torch.__version__, "cuda:", torch.cuda.is_available())' 2>&1 | tail -2
echo "shim: $(ls -l $W/src/corex_torch/_corex_shim.so 2>/dev/null || echo MISSING)"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/device_check.py" <<'PYEOF'
import sys
import traceback

sys.path.insert(0, sys.argv[1])

import torch

MISSING = []

def step(name):
    """跑一步并记录失败原因；不中断，好把缺口一次看全。"""
    def wrap(fn):
        try:
            fn()
            print(f"  [OK]   {name}")
        except Exception as exc:                        # noqa: BLE001
            first = str(exc).splitlines()[0][:160]
            print(f"  [FAIL] {name}: {type(exc).__name__}: {first}")
            MISSING.append(f"{name} -> {type(exc).__name__}: {first}")
        return fn
    return wrap

import corex_torch

print("=== 1. 注册 ===")
corex_torch.register()
print("  privateuse1 name =", torch._C._get_privateuse1_backend_name())
print("  torch.device('corex') =", torch.device("corex"))
print("  torch.cuda.is_available =", torch.cuda.is_available(), "（必须仍是 False）")
print("  codex.device_count =", torch.corex.device_count())

state = {}

@step("torch.zeros(4, device='corex')")
def _():
    state["z"] = torch.zeros(4, device="corex")
    print("        ", state["z"].device, state["z"].dtype, state["z"].shape,
          "data_ptr=0x%x" % state["z"].data_ptr())

@step("zeros 清零正确（.cpu() 读回）")
def _():
    t = state["z"].cpu()
    assert list(t) == [0.0] * 4, list(t)

@step("torch.ones(8, device='corex')")
def _():
    state["o"] = torch.ones(8, device="corex")

@step("CPU -> corex 往返逐字节相等")
def _():
    src = torch.arange(16, dtype=torch.float32) * 0.5
    dev = src.to("corex")
    back = dev.cpu()
    assert torch.equal(back, src), (back, src)
    print("        往返 16 个元素完全相等")

@step("corex 上的 a + b")
def _():
    a = torch.ones(16, device="corex")
    b = torch.ones(16, device="corex")
    s = a + b
    back = s.cpu()
    assert list(back) == [2.0] * 16, list(back)

@step(".item()")
def _():
    t = torch.ones(1, device="corex")
    v = t.item()
    assert v == 1.0, v
    print("        item =", v)

@step("非连续 strides（from_blob 造一个转置形状）")
def _():
    # 3x2 的行主序张量按 2x3 的转置 stride 读，用来确认 strides 真的被尊重。
    # dtype 必须传 ScalarType 编号（fp32 = 6）：shim 的签名是 int64_t，
    # 而 torch.dtype 没有 __int__（见 docs/00-FACTS.md §7.1 同一类坑）。
    code = corex_torch._device._scalar_type(torch.float32)
    t = corex_torch._device.shim().from_blob(
        state["o"].data_ptr(), [2, 3], [1, 2], code, 0)
    back = t.cpu()
    print("        from_blob strides OK, sum =", float(back.sum()))

@step("大张量分配 + 填 + 读回（4 MiB）")
def _():
    n = 1 << 20
    t = torch.zeros(n, dtype=torch.float32, device="corex")
    t.fill_(0.5)
    back = t.cpu()
    assert float(back.min()) == 0.5 and float(back.max()) == 0.5

@step("删张量后显存回收（empty_cache）")
def _():
    import gc
    gc.collect()
    torch.corex.empty_cache()
    print("        reserved =", torch.corex.memory_reserved() / 2**20, "MiB",
          " allocated =", torch.corex.memory_allocated() / 2**20, "MiB")

print("=== 2. 还缺的算子（这份清单就是下一步待办）===")
if MISSING:
    for line in MISSING:
        print("  -", line)
else:
    print("  (无)")
print("DEVICE_VERDICT", "PASS" if not MISSING else "PARTIAL")
PYEOF

timeout --foreground -k 5 300 "$PY" -B -u "$W/device_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
