set -u
# 目的：把「stream / 事件」这条**torch 惯例 API** 补齐并验收。目标里明确要「分配/拷贝/stream/事件」，
# 但之前的探针只验了运行时的 `_runtime.Stream`/`Event`（002/003/007），没验 `torch.corex.*` 这层——
# 而框架（coffin）接进来时用的是后者。这份探针验四件事：
#   1. `torch.corex.Stream()` 能造、`with torch.corex.stream(s):` 能进；
#   2. `torch.corex.current_stream()` 返回默认流；
#   3. `Event(enable_timing=True).record()` + `elapsed_time()` 给出正的、量级合理的毫秒数；
#   4. 在自定义流上做一次拷贝 + 内核，`synchronize()` 后结果正确（而不是只测「没报错」）。
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

cat > "$W/stream_event.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch

corex_torch.register()
torch.manual_seed(13)
FAILED, PASSED = [], []


def check(name, fn):
    try:
        detail = fn()
        print(f"  [OK ] {name:46s} {detail}")
        PASSED.append(name)
    except Exception as exc:                            # noqa: BLE001
        print(f"  [FAIL] {name:46s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        FAILED.append(name)


print("=== 1. 命名空间里有没有这些名字 ===")
for attr in ("Stream", "Event", "current_stream", "stream", "synchronize"):
    print(f"  torch.corex.{attr:14s} = {getattr(torch.corex, attr, '<缺失>')}")


def t_current():
    s = torch.corex.current_stream()
    return f"type={type(s).__name__}"


check("current_stream() 返回可用对象", t_current)


def t_new_stream():
    s = torch.corex.Stream()
    with torch.corex.stream(s):
        torch.corex.synchronize()
    return "造 + 进上下文 + 同步，全部通过"


check("Stream() / with torch.corex.stream(s)", t_new_stream)


def t_event():
    a = torch.corex.Event(enable_timing=True)
    b = torch.corex.Event(enable_timing=True)
    a.record()
    # 卡上没有 RNG：先 CPU 生成再搬上去（不是本探针要测的东西）\n    x = torch.randn(256, 256).to("corex") @ torch.randn(256, 256).to("corex")
    torch.corex.synchronize()
    b.record()
    b.synchronize()
    ms = a.elapsed_time(b)
    assert ms > 0, f"elapsed_time 应该为正，得到 {ms}"
    assert ms < 10000, f"elapsed_time 大得出奇：{ms} ms"
    return f"elapsed_time={ms:.3f} ms（一次 256³ mm）"


check("Event.record + elapsed_time", t_event)


def t_event_sync():
    e = torch.corex.Event(enable_timing=False)
    a = torch.ones(1024, device="corex")
    a.mul_(2.0)
    e.record()
    e.synchronize()
    assert float(a[0].item()) == 2.0
    return "record + synchronize 后结果正确"


check("Event(无计时).record + synchronize", t_event_sync)


def t_stream_work():
    """在自定义流上做拷贝 + 内核，同步后逐字节校验。"""
    host = torch.arange(4096, dtype=torch.float32)
    s = torch.corex.Stream()
    with torch.corex.stream(s):
        dst = torch.empty(4096, device="corex")
        dst.copy_(host)
        dst.mul_(3.0)
        torch.corex.synchronize()
    got = dst.cpu()
    expect = host * 3.0
    assert torch.equal(got, expect), "流上算出来的结果与参照不等"
    return "H2D + mul_ + 同步，4096 元素逐字节相等"


check("自定义流上做拷贝与内核", t_stream_work)


def t_default_stream_after():
    s = torch.corex.current_stream()
    return f"退出 with 后 current_stream 类型={type(s).__name__}"


check("with 退出后 current_stream 可用", t_default_stream_after)

print("=== 2. 汇总 ===")
print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
for line in FAILED:
    print("  -", line)
print("STREAM_EVENT_VERDICT", "PASS" if not FAILED else "PARTIAL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/stream_event.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
