set -u
# 目的：用**链接判据**（llc + lld --no-undefined）判定 bf16 向量转换的三条候选路，纠正 probes/194
# 的误判——194 只看 `llc -filetype=obj` 返回码，而 LLVM 允许把**未实现的外部函数调用**留在目标
# 文件里（llc 照样 rc=0），真正拦住它的是 corex-triton make_cubin 里的 `lld --no-undefined`。
# 所以 194 得出的「@llvm.bi.bfloat2float.v8bf16 可用」是错的：192 真机跑出来是
#     lld: error: undefined symbol: llvm.bi.bfloat2float.v8bf16
#
# 本探针拿 probes/193 落盘的真实 biir 做三组对照，三组**都跑到链接为止**：
#   V1 原样 fpext >8 x bfloat>              → 预期 llc 就拒（基线）
#   V2 过一遍 _bf16.rewrite_bf16_conversions → 预期 llc 过、**lld 挂**（复现 192 的现象）
#   V3 V2 + 把向量 intrinsic 调用**标量化**（extractelement → 标量 call → insertelement）
#                                          → 预期 llc 过、lld 也过（这是给 _bf16 的修法）
#
# 依赖：先跑 probes/193（落盘 diag_biir__row_reduce_kernel.ll）。
# 跑法：python tools/remote.py sbx-run probes/195_bf16_link_verdict.sh --name 195_bf16_link --timeout 600

PY=$HOME/venvs/triton34/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
CX=/usr/local/corex
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
export TB_W=$W
export PYTHONPATH=$W/corex-torch/src
mkdir -p "$W"

echo "HOST=$(hostname) DATE=$(date -Is)"
echo "=== 0. 找 probes/193 落盘的那份 biir ==="
D=$(ls -t "$W"/diag_biir*.ll 2>/dev/null | head -1)
if [ -z "${D:-}" ]; then echo "NO_DUMP（先跑 probes/193）"; exit 66; fi
echo "DUMP=$D"

cat > "$W/link_verdict.py" <<'PYEOF'
"""llc + lld --no-undefined 判定 bf16 向量转换的三条路。"""

import glob
import os
import re
import subprocess

W = os.environ["TB_W"]
LLC = "/usr/local/corex/bin/llc"
LLD = "/usr/local/corex/bin/lld"
LLC_ARGS = [LLC, "-march=bi", "-mcpu=ivcore10", "-mtriple=bi-iluvatar-ilurt",
            "-filetype=obj", "-O3"]
LLD_ARGS = [LLD, "-flavor", "ld.lld", "--no-warn-missing-entry", "--no-undefined"]

# 向量 intrinsic 调用（_bf16 上完向量正则之后的形态）
VEC_WIDE = re.compile(
    r"(?P<lhs>%[-\w.$]+)\s*=\s*call\s+<(?P<n>\d+) x float>\s+"
    r"@llvm\.bi\.bfloat2float\.v(?P<n2>\d+)bf16\(<(?P=n) x bfloat>\s+(?P<arg>%[-\w.$]+)\)")
VEC_NARROW = re.compile(
    r"(?P<lhs>%[-\w.$]+)\s*=\s*call\s+<(?P<n>\d+) x bfloat>\s+"
    r"@llvm\.bi\.float2bfloat\.v(?P<n2>\d+)bf16\(<(?P=n) x float>\s+(?P<arg>%[-\w.$]+)\)")


def scalarize(ir):
    """把向量 intrinsic 调用拆成逐元素 extract → 标量 call → insert 链。"""
    total = 0

    def wide(m):
        nonlocal total
        total += 1
        n, lhs, arg = int(m.group("n")), m.group("lhs"), m.group("arg")
        out, prev = [], None
        for i in range(n):
            out.append(f"  %zzwe{i} = extractelement <{n} x bfloat> {arg}, i32 {i}")
            out.append(f"  %zzwc{i} = call float @llvm.bi.bfloat2float(bfloat %zzwe{i})")
            dst = lhs if i == n - 1 else f"%zzwi{i}"
            src = "poison" if prev is None else prev
            out.append(f"  {dst} = insertelement <{n} x float> {src}, "
                       f"float %zzwc{i}, i32 {i}")
            prev = dst
        return "\n".join(out)

    def narrow(m):
        nonlocal total
        total += 1
        n, lhs, arg = int(m.group("n")), m.group("lhs"), m.group("arg")
        out, prev = [], None
        for i in range(n):
            out.append(f"  %zzne{i} = extractelement <{n} x float> {arg}, i32 {i}")
            out.append(f"  %zznc{i} = call bfloat @llvm.bi.float2bfloat(float %zzne{i})")
            dst = lhs if i == n - 1 else f"%zzni{i}"
            src = "poison" if prev is None else prev
            out.append(f"  {dst} = insertelement <{n} x bfloat> {src}, "
                       f"bfloat %zznc{i}, i32 {i}")
            prev = dst
        return "\n".join(out)

    ir = VEC_WIDE.sub(wide, ir)
    ir = VEC_NARROW.sub(narrow, ir)
    return ir, total


def build_and_link(tag, ir, results):
    """写 .ll → llc → lld，两级都记录。"""
    path = os.path.join(W, "lv_%s.ll" % tag)
    open(path, "w", encoding="utf-8").write(ir)
    obj = path + ".o"
    p1 = subprocess.run(LLC_ARGS + [path, "-o", obj], capture_output=True, text=True)
    if p1.returncode != 0:
        msg = (p1.stderr or p1.stdout or "").strip().splitlines()
        print("  %-40s llc=FAIL  %s" % (tag, msg[0][:150] if msg else ""))
        results[tag] = ("llc", p1.returncode)
        return
    p2 = subprocess.run(LLD_ARGS + [obj, "-o", obj + ".elf"], capture_output=True, text=True)
    if p2.returncode != 0:
        msg = (p2.stderr or p2.stdout or "").strip().splitlines()
        print("  %-40s llc=OK lld=FAIL  %s" % (tag, msg[0][:150] if msg else ""))
        results[tag] = ("lld", p2.returncode)
        return
    size = os.path.getsize(obj + ".elf") if os.path.exists(obj + ".elf") else 0
    print("  %-40s llc=OK lld=OK  elf=%6d B" % (tag, size))
    results[tag] = ("ok", 0)


def main():
    dump = sorted(glob.glob(os.path.join(W, "diag_biir*.ll")), key=os.path.getmtime)[-1]
    raw = open(dump, encoding="utf-8", errors="replace").read()
    print("dump =", os.path.basename(dump), " 行数 =", len(raw.splitlines()))

    from corex_torch import _bf16
    rewritten, hits = _bf16.rewrite_bf16_conversions(raw)
    print("rewrite_bf16_conversions 命中 =", hits,
          " 改写后剩余 fpext =", rewritten.count("fpext"),
          " fptrunc =", rewritten.count("fptrunc"),
          " .v*bf16 调用 =", len(VEC_WIDE.findall(rewritten)) + len(VEC_NARROW.findall(rewritten)))

    scal, n_scal = scalarize(rewritten)
    print("标量化：处理了", n_scal, "处向量 intrinsic 调用")

    print("=== 三级判定（llc → lld --no-undefined）===")
    results = {}
    build_and_link("V1_原样_vector_fpext", raw, results)
    build_and_link("V2_过改写器_保留向量intrinsic", rewritten, results)
    build_and_link("V3_过改写器_再标量化", scal, results)

    print("=== 结论 ===")
    v1 = results.get("V1_原样_vector_fpext", ("", 1))[0]
    v2 = results.get("V2_过改写器_保留向量intrinsic", ("", 1))[0]
    v3 = results.get("V3_过改写器_再标量化", ("", 1))[0]
    print("  V1 =", v1, " V2 =", v2, " V3 =", v3)
    if v3 == "ok":
        print("  → 标量化可行：_bf16.py 应把向量 fpext/fptrunc 展开成 extract/call/insert 链")
    elif v2 == "ok":
        print("  → 向量 intrinsic 本身可用，192 的 lld 失败另有原因")
    else:
        print("  → 连标量化都过不了链接，需要彻底换策略")
    print("LINKVERDICT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 1. 链接判据三级对照 ==="
timeout --foreground -k 5 240 "$PY" -B -u "$W/link_verdict.py"
PYRC=$?
echo "PY_RC=$PYRC"
echo "PROBE195_DONE py_rc=$PYRC"
exit "$PYRC"
