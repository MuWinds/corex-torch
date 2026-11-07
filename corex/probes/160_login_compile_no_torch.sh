set -u
# 目的（第 2 步，登录节点侧）：证明「编译侧不需要 torch」这条链在 import 层就成立。
#
# 要回答的三个具体问题：
#   1) `import triton_corex` 会不会因为 driver 模块顶层碰 torch 而失败？
#      （看 src/triton_corex/driver.py：所有 torch 都在函数体内，没有模块级 import）
#   2) 完全不 import torch 的进程里，triton.compile 能走多远？
#   3) 这个进程里第一次 import torch 到底发生在哪一步？
#
# 为什么只能在登录节点跑到「cubin 之前」：登录节点没有 /usr/local/corex
# （corex-torch/AGENTS.md §3），最后的 cubin 阶段要调厂商 llc，必然失败。所以本脚本
# 判定到 biir 为止；cubin + 启动留给沙箱里的 151_torchfree_launch.sh。
#
# 跑法（登录节点，不走沙箱队列，不碰卡）：
#   python tools/remote.py put corex-torch/probes/160_login_compile_no_torch.sh \
#       ~/coffin-work/base/sandbox/work/tb/160_login_compile_no_torch.sh
#   python tools/remote.py sh 'bash ~/coffin-work/base/sandbox/work/tb/160_login_compile_no_torch.sh'
#
# 载荷：$W/tb/tb_src.tgz（tar czf tb_src.tgz -C corex-triton/src triton_corex）
# 判定：PROBE160_VERDICT PASS。

PY_TRITON=${HOME}/venvs/triton34/bin/python
S=${HOME}/coffin-work/base/sandbox
W=$S/work
T=$W/tb
# 解包到独立的目录：151 在沙箱里会 rm -rf $T/src，两边同时解包会互相踩。
export TB_W160=$T
export COREX_DUMP_IR=$T/login_biir.ll

echo "HOST=$(hostname) DATE=$(date -Is)  # 登录节点没有 /usr/local/corex，见下"

mkdir -p "$T"
[ -x "$PY_TRITON" ] || { echo "MISSING_PY $PY_TRITON"; exit 66; }
if [ -f "$T/tb_src.tgz" ]; then
  rm -rf "$T/login_src" && mkdir -p "$T/login_src" && tar xzf "$T/tb_src.tgz" -C "$T/login_src" || exit 67
else
  echo "PAYLOAD_MISSING $T/tb_src.tgz"; exit 66
fi
echo "CORE_HOME_EXISTS=$([ -d /usr/local/corex ] && echo yes || echo no)"

cat > "$T/160_compile.py" <<'PYEOF'
"""登录节点侧的第 2 步探针：import 链有没有把 torch 拖进来，编译能走到哪个阶段。"""

import os
import sys

T = os.environ["TB_W160"]
sys.path.insert(0, os.path.join(T, "login_src"))


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def say(tag):
    mods = torchmods()
    print("[%s] torch_modules=%d %s" % (tag, len(mods), mods[:8]))


def main():
    say("start")
    import triton
    print("triton_version=%s" % triton.__version__)
    print("triton_path=%s" % triton.__file__)
    say("after_import_triton")

    import triton_corex
    import triton.language as tl
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    print("backends=%s" % sorted(backends))
    print("corex_registered=%s" % ("corex" in backends))
    say("after_import_triton_corex")

    @triton.jit
    def vecadd(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(c_ptr + off, tl.load(a_ptr + off, mask=mask) + tl.load(b_ptr + off, mask=mask),
                 mask=mask)

    src = triton.compiler.ASTSource(
        fn=vecadd,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "n": "i32",
                   "BLOCK": "constexpr"},
        constexprs={"BLOCK": 128},
    )
    err = None
    try:
        kernel = triton.compile(src, target=GPUTarget("corex", 70, 32))
        print("COMPILE_OK asm_keys=%s" % sorted(kernel.asm.keys()))
        print("cubin_bytes=%d" % len(kernel.asm["cubin"]))
    except BaseException as exc:                          # noqa: BLE001
        err = exc
        print("COMPILE_RAISED %s: %s" % (type(exc).__name__, str(exc)[:300]))
    say("after_compile")

    # 改写产物在 make_cubin 的第一行就落盘（src/triton_corex/compiler.py:259-265），
    # 所以 llc 缺失也不影响拿到 biir —— 这一点正好用来在登录节点验证改写链。
    dump = os.environ.get("COREX_DUMP_IR", "")
    mt, bi, triple = -1, -1, ""
    if dump and os.path.exists(dump):
        text = open(dump, "r", encoding="utf-8", errors="replace").read()
        mt = text.count("llvm.bi.")
        bi = text.count("iluvatar_kernel")
        triple = next((ln.strip() for ln in text.splitlines() if "target triple" in ln), "")
        print("biir_dump bytes=%d llvm_bi_hits=%d iluvatar_kernel_hits=%d" % (len(text), mt, bi))
        print("biir_triple=%s" % triple)
    else:
        print("biir_dump MISSING（改写没跑到）")

    # 判定：全程 0 个 torch 模块、且 biir 真的产出了、且失败点正是缺 llc。
    no_torch = len(torchmods()) == 0
    has_biir = mt > 0 and bi > 0
    at_llc = err is not None and "llc" in str(err)
    print("no_torch=%s has_biir=%s failed_at_llc=%s" % (no_torch, has_biir, at_llc))
    print("PROBE160_VERDICT", "PASS" if (no_torch and has_biir) else "FAIL")
    return 0 if (no_torch and has_biir) else 7


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

cat > "$T/160b_first_torch.py" <<'PYEOF'
"""第 2 步的边界问题：这个进程里第一次 import torch 发生在哪一步？

结论预期（读 triton/backends/driver.py:37-49）：GPUDriver.__init__ 里 `import torch`，
而 CorexDriver.__init__ 会 super().__init__()，所以「构造 driver」是第一个 torch 触点。
本探针把它单独跑一遍，和编译主流程分开，免得失败污染主结论。
"""

import os
import sys

T = os.environ["TB_W160"]
sys.path.insert(0, os.path.join(T, "login_src"))


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


import triton_corex  # noqa: E402
from triton.runtime.driver import driver as drv  # noqa: E402

print("[before] torch_modules=%d" % len(torchmods()))
try:
    triton_corex.set_active()
    print("SET_ACTIVE_OK")
except BaseException as exc:                             # noqa: BLE001
    print("SET_ACTIVE_RAISED %s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:200]))
mods = torchmods()
print("[after] torch_modules=%d %s" % (len(mods), mods[:6]))
print("driver_active_is_proxy=%s" % type(drv.active).__name__)
print("PROBE160B_DONE")
PYEOF

echo "=== 160: import 链 + 编译到 biir ==="
timeout --foreground -k 5 180 "$PY_TRITON" -B -u "$T/160_compile.py"
RC=$?
echo "PY_RC=$RC"

echo "=== 160b: 第一次 import torch 发生在哪一步 ==="
timeout --foreground -k 5 120 "$PY_TRITON" -B -u "$T/160b_first_torch.py"
RC2=$?
echo "PY2_RC=$RC2"

echo "=== biir 头部 6 行 ==="
head -6 "$T/login_biir.ll" 2>/dev/null || echo "NO_DUMP"

echo "PROBE160_DONE rc=$RC rc2=$RC2"
exit "$RC"
