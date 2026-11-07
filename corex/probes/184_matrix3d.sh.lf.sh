set -u
# 目的：一条龙验证（部署包 → 矩阵 v3 → 22 条功能矩阵）。
# 之前几版探针是「按名字替换」改出来的，tar 名换错一次就白跑一轮；这里写死当前包名。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorexi.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"
echo "部署完成：$(ls "$SITE/corex" | wc -l) 个文件；topk 实现存在=$(grep -c _topk_impl "$SITE/corex/_aten_kernels.py")"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
echo "=== 矩阵 v3 ==="
( cd "$H/work" && "$PY" -u opsmatrix3.py 2>&1 | grep -E "^(RUN|FAIL|MISMATCH|SUMMARY)" | tail -28 )
echo "T184_DONE"
