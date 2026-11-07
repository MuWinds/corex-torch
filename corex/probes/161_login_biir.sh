set -u
# 目的（第 2 步，登录节点侧，修正版）：证明「`import triton_corex` + `triton.compile` 一路到
# biir 全程不需要 torch」，并取回改写后的 IR 作为证据。
#
# 为什么要有这一条（161）而不是改 160：160 第一版判定本脚本时，把
# `COREX_DUMP_IR=<路径>` 的落盘文件名猜成了 `<路径>` 本身。实际实现是
# `src/triton_corex/compiler.py:259-265`：
#     stem.with_name(f"{stem.stem}_{kn}{stem.suffix}")
# 也就是**会在文件名里插入 kernel 名**（实测落成 `login_biir_vecadd.ll`）。
# 于是 160 报 `biir_dump MISSING` → 判定 FAIL，而链路本身是好的。160 保留原样作为
# 「探针自己对 dump 文件名的假设会骗人」的证据，本文件用正确判据重跑。
#
# 另外 160 的判据还错在第二条：它用 `llvm.bi.` 出现次数 > 0 当作「改写成功」，
# 而 vecadd 里根本不需要任何 warp 级原语，`llvm.bi.` 计数必然是 0。正确判据是
# triple 换成 `bi-iluvatar-ilurt` 且出现 `define iluvatar_kernel`。
#
# 为什么只能在登录节点跑到 biir：登录节点没有 /usr/local/corex（docs/00-FACTS.md §2），
# cubin 阶段要调厂商 llc，必然失败。cubin + 启动见沙箱里的 151。
#
# 跑法（登录节点，不走沙箱队列）：
#   python tools/remote.py put corex-torch/probes/161_login_biir.sh \
#       ~/coffin-work/base/sandbox/work/tb/161_login_biir.sh
#   python tools/remote.py sh 'bash ~/coffin-work/base/sandbox/work/tb/161_login_biir.sh'
# 判定：PROBE161_VERDICT PASS。

PY_TRITON=${HOME}/venvs/triton34/bin/python
S=${HOME}/coffin-work/base/sandbox
W=$S/work
T=$W/tb
export TB_W161=$T
export COREX_DUMP_IR=$T/biir161.ll

echo "HOST=$(hostname) DATE=$(date -Is)"
echo "CORE_HOME_EXISTS=$([ -d /usr/local/corex ] && echo yes || echo no)  # 登录节点预期 no"

mkdir -p "$T"
[ -x "$PY_TRITON" ] || { echo "MISSING_PY $PY_TRITON"; exit 66; }
if [ -f "$T/tb_src.tgz" ]; then
  rm -rf "$T/login_src" && mkdir -p "$T/login_src" && tar xzf "$T/tb_src.tgz" -C "$T/login_src" || exit 67
else
  echo "PAYLOAD_MISSING $T/tb_src.tgz"; exit 66
fi

cat > "$T/161_biir.py" <<'PYEOF'
"""登录节点：无 torch 进程里把 vecadd 编到 biir，并核对改写产物。"""

import glob
import os
import sys

T = os.environ["TB_W161"]
sys.path.insert(0, os.path.join(T, "login_src"))


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def say(tag):
    mods = torchmods()
    print("[%s] torch_modules=%d %s" % (tag, len(mods), mods[:8]))


def main():
    say("start")
    import triton
    import triton.language as tl
    import triton_corex  # noqa: F401
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    print("triton=%s backends=%s corex_registered=%s"
          % (triton.__version__, sorted(backends), "corex" in backends))
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
        print("COMPILE_OK（登录节点不该走到这里：缺 llc）asm=%s" % sorted(kernel.asm.keys()))
    except BaseException as exc:                          # noqa: BLE001
        err = exc
        print("COMPILE_RAISED %s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:200]))
    say("after_compile")

    # 正确判据：dump 会在文件名里插 kernel 名（compiler.py:259-265）。
    pat = os.path.splitext(os.environ["COREX_DUMP_IR"])[0] + "_*.ll"
    hits = sorted(glob.glob(pat))
    print("dump_glob=%s hits=%s" % (pat, [os.path.basename(h) for h in hits]))
    ok_dump = False
    if hits:
        text = open(hits[0], "r", encoding="utf-8", errors="replace").read()
        triple_ok = 'target triple = "bi-iluvatar-ilurt"' in text
        kernel_marker = text.count("define iluvatar_kernel")
        bi_hits = text.count("llvm.bi.")
        nvvm_sreg = text.count("llvm.nvvm.read.ptx.sreg")
        params = 0
        for ln in text.splitlines():
            if "define iluvatar_kernel" in ln and "@" in ln:
                params = ln.count(",") + 1
                print("define_line=%s" % ln.strip()[:200])
        print("biir bytes=%d triple_bi=%s iluvatar_kernel=%d llvm_bi_hits=%d nvvm_sreg_hits=%d params=%d"
              % (len(text), triple_ok, kernel_marker, bi_hits, nvvm_sreg, params))
        # vecadd 不含 warp 级原语，所以 llvm.bi.* 为 0 是正常的；
        # 判定只看「triple 改了 + kernel 标记加了 + 线程号已换成 NVVM sreg」。
        ok_dump = triple_ok and kernel_marker == 1 and nvvm_sreg > 0
    at_llc = err is not None and "llc" in str(err)
    print("no_torch=%s dump_ok=%s failed_at_llc=%s"
          % (len(torchmods()) == 0, ok_dump, at_llc))
    verdict = "PASS" if (len(torchmods()) == 0 and ok_dump and at_llc) else "FAIL"
    print("PROBE161_VERDICT", verdict)
    return 0 if verdict == "PASS" else 7


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 161: 无 torch 编到 biir ==="
timeout --foreground -k 5 180 "$PY_TRITON" -B -u "$T/161_biir.py"
RC=$?
echo "PY_RC=$RC"

echo "=== 事实现场：160 报 MISSING 的那个名字 vs 真正的落盘名 ==="
ls -l "$T"/login_biir*.ll "$T"/biir161*.ll 2>/dev/null

echo "PROBE161_DONE rc=$RC"
exit "$RC"
