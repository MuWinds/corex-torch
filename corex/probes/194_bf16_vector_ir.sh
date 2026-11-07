set -u
# 目的：`_bf16.rewrite_bf16_conversions` 只认**标量**转换（`fpext bfloat %x to float`），不认
# **向量**形式（`fpext <8 x bfloat> %x to <8 x float>`），而规约/softmax/norm/mm 的内核里恰好是
# 向量形式（`probes/193` 抓到原文：`%63 = fpext <8 x bfloat> %62 to <8 x float>`）。
#
# 本探针不再猜：直接拿 probes/193 落盘的**真实 biir**，把那一行换成候选形式交给 llc 判：
#   1. `@llvm.bi.bfloat2float` 有没有**向量重载**（两种拼写都试）；
#   2. 没有的话，把这一行**手工标量化**（extractelement → 标量 call → insertelement）能不能过；
#   3. 反方向（`<N x float>` → `<N x bfloat>`）同样问一遍（用 alloca+store 保活，避免被 DCE）。
# 结论直接决定 `_bf16` 该「加向量重载」还是「自动标量化」。
#
# 依赖：先跑过 probes/193_bf16_reduce_diag.sh（它落盘 diag_biir__row_reduce_kernel.ll）。
# 跑法：python tools/remote.py sbx-run probes/194_bf16_vector_ir.sh --name 194_bf16_vec --timeout 600

PY=$HOME/venvs/triton34/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
CX=/usr/local/corex
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
export TB_W=$W
mkdir -p "$W"

echo "HOST=$(hostname) DATE=$(date -Is)"
echo "=== 0. 找 probes/193 落盘的那份 biir ==="
D=$(ls -t "$W"/diag_biir*.ll 2>/dev/null | head -1)
if [ -z "${D:-}" ]; then echo "NO_DUMP（先跑 probes/193）"; exit 66; fi
echo "DUMP=$D"

cat > "$W/vec_ir.py" <<'PYEOF'
"""把真实 biir 里的向量 fpext / fptrunc 换成候选形式，交给 llc 判能不能编。"""

import glob
import os
import re
import subprocess

W = os.environ["TB_W"]
LLC = "/usr/local/corex/bin/llc"
ARGS = [LLC, "-march=bi", "-mcpu=ivcore10", "-mtriple=bi-iluvatar-ilurt",
        "-filetype=obj", "-O3"]
DECL_S = ("declare float @llvm.bi.bfloat2float(bfloat)\n"
          "declare bfloat @llvm.bi.float2bfloat(float)\n")


def run_llc(path, tag, results):
    out = path + ".o"
    proc = subprocess.run(ARGS + [path, "-o", out], capture_output=True, text=True)
    first = ""
    if proc.returncode != 0:
        txt = (proc.stderr or proc.stdout or "").strip()
        first = txt.splitlines()[0][:200] if txt else "(no output)"
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print("  %-46s rc=%d obj=%6d B %s" % (tag, proc.returncode, size, first))
    results[tag] = proc.returncode
    return proc.returncode


def main():
    dump = sorted(glob.glob(os.path.join(W, "diag_biir*.ll")), key=os.path.getmtime)[-1]
    lines = open(dump, encoding="utf-8", errors="replace").read().splitlines()

    pat = re.compile(r"(?P<lhs>%[-\w.$]+)\s*=\s*fpext\s+(?P<vt><\d+ x bfloat>)\s+"
                     r"(?P<arg>%[-\w.$]+)\s+to\s+(?P<ft><\d+ x float>)")
    idx = match = None
    for i, ln in enumerate(lines):
        m = pat.search(ln)
        if m:
            idx, match = i, m
            break
    if idx is None:
        print("dump 里没有向量 fpext 行，无法做这个实验")
        return 2
    lhs = match.group("lhs")
    vt = match.group("vt")
    arg = match.group("arg")
    ft = match.group("ft")
    N = int(re.search(r"<(\d+) x bfloat>", vt).group(1))
    suffix = ", !dbg !21" if "!dbg" in lines[idx] else ""
    print("=== 目标行（第 %d 行）: %s" % (idx + 1, lines[idx].strip()))
    print("    lhs=%s vt=%s arg=%s ft=%s N=%d" % (lhs, vt, arg, ft, N))

    counter = [0]

    def variant(new_lines, decl_extra=""):
        body = lines[:idx] + new_lines + lines[idx + 1:]
        txt = "\n".join(body) + "\n"
        if decl_extra and decl_extra.strip() not in txt:
            txt += decl_extra
        counter[0] += 1
        path = os.path.join(W, "vari_%03d.ll" % counter[0])
        open(path, "w", encoding="utf-8").write(txt)
        return path

    results = {}
    print("=== §1 基线：原样（预期 llc 拒绝）===")
    run_llc(variant([lines[idx]]), "A0 原样 vector fpext", results)

    print("=== §2 bfloat2float 的向量重载（两种拼写）===")
    run_llc(variant(["  %s = call %s @llvm.bi.bfloat2float(%s %s)%s"
                     % (lhs, ft, vt, arg, suffix)],
                    "declare %s @llvm.bi.bfloat2float(%s)\n" % (ft, vt)),
            "A1 @llvm.bi.bfloat2float(<N x bfloat>)", results)
    run_llc(variant(["  %s = call %s @llvm.bi.bfloat2float.v%dbf16(%s %s)%s"
                     % (lhs, ft, N, vt, arg, suffix)],
                    "declare %s @llvm.bi.bfloat2float.v%dbf16(%s)\n" % (ft, N, vt)),
            "A2 @llvm.bi.bfloat2float.v{N}bf16", results)

    print("=== §3 手工标量化（extract → 标量 call → insert）===")
    chain, prev = [], "poison"
    for i in range(N):
        chain.append("  %%ve%d = extractelement %s %s, i32 %d" % (i, vt, arg, i))
        chain.append("  %%vc%d = call float @llvm.bi.bfloat2float(bfloat %%ve%d)" % (i, i))
        if i == N - 1:
            chain.append("  %s = insertelement %s %s, float %%vc%d, i32 %d"
                         % (lhs, ft, prev, i, i))
        else:
            chain.append("  %%vi%d = insertelement %s %s, float %%vc%d, i32 %d"
                         % (i, ft, prev, i, i))
            prev = "%%vi%d" % i
    run_llc(variant(chain, DECL_S), "A3 标量化（%d 元素）" % N, results)

    print("=== §4 反方向 <N x float> → <N x bfloat>（写回时用；alloca+store 保活）===")
    # 注意：必须**继续用 %63 这个名字**——LLVM 文本 IR 要求无名临时量按顺序编号，删掉 %63 的
    # 定义后面会报 "instruction expected to be numbered"。这是实验脚本的约束，不是后端行为。
    names = [lhs] + ["%%znv%d" % i for i in range(1, N)]
    lead = ["  %s = insertelement %s poison, float 0.0, i32 0" % (names[0], ft)]
    for i in range(1, N):
        lead.append("  %s = insertelement %s %s, float 0.0, i32 %d"
                    % (names[i], ft, names[i - 1], i))
    last = names[-1]
    for tag, decl, call in (
        ("B1 @llvm.bi.float2bfloat(<N x float>)",
         "declare %s @llvm.bi.float2bfloat(%s)\n" % (vt, ft),
         "  %%zn = call %s @llvm.bi.float2bfloat(%s %s)" % (vt, ft, last)),
        ("B2 @llvm.bi.float2bfloat.v{N}bf16",
         "declare %s @llvm.bi.float2bfloat.v%dbf16(%s)\n" % (vt, N, ft),
         "  %%zn = call %s @llvm.bi.float2bfloat.v%dbf16(%s %s)" % (vt, N, ft, last)),
    ):
        # 不能用 alloca（内核里是 dynamic_stackalloc，bi 后端 selects 不出来——那会掩盖真正要测的
        # 指令能不能选）。改成 store 到函数第一个指针参数上，纯为保活。
        body = lead + [call, "  store %s %%zn, ptr addrspace(1) %%0, align 16" % vt]
        run_llc(variant(body, decl), tag, results)

    print("=== §5 结论 ===")
    ok_vec = (results.get("A1 @llvm.bi.bfloat2float(<N x bfloat>)", 1) == 0
              or results.get("A2 @llvm.bi.bfloat2float.v{N}bf16", 1) == 0)
    ok_scal = results.get("A3 标量化（%d 元素）" % N, 1) == 0
    ok_rev = (results.get("B1 @llvm.bi.float2bfloat(<N x float>)", 1) == 0
              or results.get("B2 @llvm.bi.float2bfloat.v{N}bf16", 1) == 0)
    print("  向量重载可用=%s   标量化可用=%s   反方向向量可用=%s" % (ok_vec, ok_scal, ok_rev))
    if ok_vec:
        print("  → _bf16 只需把正则从标量扩到向量，直接换向量 intrinsic")
    elif ok_scal:
        print("  → _bf16 需要自动标量化（extract/call/insert 链）才能覆盖向量转换")
    else:
        print("  → 两条都不通：向量 bf16 转换没有可用形态，要回报 Lead")
    print("VECIR_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 1. 向量形式 IR 实验 ==="
timeout --foreground -k 5 240 "$PY" -B -u "$W/vec_ir.py"
PYRC=$?
echo "PY_RC=$PYRC"
echo "PROBE194_DONE py_rc=$PYRC"
exit "$PYRC"
