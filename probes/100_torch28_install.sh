set -u
# 探针 100：在登录节点把上游 CPU 版 torch 2.8.0 装进隔离 venv ~/venvs/corex28，
# 并验证它真的能 import。
#
# 为什么在登录节点做：装 torch 是纯 CPU、纯网络的事，碰不到卡，也不必占串行沙箱队列。
# 为什么必须隔离 venv：共享 py310 里的 torch 2.1.0+corex.3.2.1 是全环境数值基线，
# 不能被覆盖（AGENTS.md §4）。本脚本只动 ~/venvs/corex28 与 ~/wheels，不碰共享环境。
#
# 两次踩到的坑（都有实测依据，别重复踩）：
#  1) venv 自带的 pip 23.0.1 太老：下载 typing_extensions-4.16.0 时把
#     `typing-extensions` 与 `typing_extensions` 判成两个名字（inconsistent Name），
#     丢掉 wheel 退化成源码构建，然后卡在 flit_core 上失败。
#     证据：~/venvs/corex28-pip.log。→ 先把 pip 升上来（清华镜像）。
#  2) 升级后的 pip 直连 download.pytorch.org 下 184 MB 的 wheel 只有 ~50 kB/s
#     （与同时段 curl 的 7.1 MB/s 差两个数量级），实测 10 分钟只走了 28 MB。
#     → 用 curl（带断点续传与重试）只下 wheel 文件，再用 pip 装本地文件。
#     curl 25 s 下完 184 021 630 B，sha256 与 index 页一致。
#
# 用法（登录节点直接跑，不要投沙箱队列）：
#   python tools/remote.py put probes/100_torch28_install.sh /data/home/2024240384035/p100.sh
#   python tools/remote.py sh 'bash /data/home/2024240384035/p100.sh > /data/home/2024240384035/p100.out 2>&1'
# 期望最后一行 TORCH28_VERDICT PASS。

set -e
HOME_DIR=/data/home/2024240384035
VENV=$HOME_DIR/venvs/corex28
PY=$VENV/bin/python
WHEELS=$HOME_DIR/wheels
WHEEL=torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
# 官方 index 页 https://download.pytorch.org/whl/cpu/torch/ 里该文件后面挂的 sha256
SHA_EXPECT=16d75fa4e96ea28a785dfd66083ca55eb1058b6d6c5413f01656ca965ee2077e
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple
CPU_URL="https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"
LOG=$HOME_DIR/venvs/corex28-pip100.log

echo "=== 0. 现状 $(date -Is) ==="
uname -srm
echo "venv 是否已存在: $(test -d "$VENV" && echo yes || echo no)"
test -f "$VENV/pyvenv.cfg" && cat "$VENV/pyvenv.cfg"
if [ ! -x "$PY" ]; then
    # 只在 venv 缺失时创建；已存在的一律复用，避免动到已经装好的东西。
    echo "STEP0 建 venv"
    /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -m venv "$VENV"
fi
"$PY" -V
"$PY" -m pip -V

echo "=== 1. 升级 pip（清华镜像；旧 pip 23.0.1 有名字规范化 bug）$(date -Is) ==="
"$PY" -m pip install -q -U pip -i "$TUNA"
"$PY" -m pip -V

echo "=== 2. 先装 torch 的纯 Python 依赖（清华镜像）$(date -Is) ==="
# 单独装依赖是为了让第 4 步可以 --no-deps --no-index，完全不联网解析：
# 依赖已满足时 pip 不会再去构建 typing-extensions 那几个纯 Python 包。
"$PY" -m pip install -q -i "$TUNA" \
    filelock jinja2 networkx sympy fsspec typing-extensions numpy
"$PY" -m pip list 2>/dev/null | grep -iE '^(numpy|sympy|networkx|jinja2|filelock|fsspec|typing_extensions) '

echo "=== 3. 备好 torch wheel（curl 直下 + 断点续传 + sha256 校验）$(date -Is) ==="
mkdir -p "$WHEELS"
if [ -f "$WHEELS/$WHEEL" ] && echo "$SHA_EXPECT  $WHEELS/$WHEEL" | sha256sum -c - >/dev/null 2>&1; then
    echo "STEP3 已存在且 sha256 匹配，跳过下载"
else
    # -C - 断点续传：上一次 curl 留下的半截文件能接着用，不用从头再来。
    curl -L -C - --retry 30 --retry-all-errors --retry-delay 3 \
         --connect-timeout 20 -sS -o "$WHEELS/$WHEEL" "$CPU_URL"
    echo "$SHA_EXPECT  $WHEELS/$WHEEL" | sha256sum -c - || {
        echo "STEP3_SHA_MISMATCH"; exit 3; }
fi
ls -l "$WHEELS/$WHEEL"

echo "=== 4. 装 torch 2.8.0+cpu（本地 wheel，不联网）$(date -Is) ==="
"$PY" -m pip install --no-deps --no-index "$WHEELS/$WHEEL" > "$LOG" 2>&1 || {
    echo "STEP4_FAILED tail:"; tail -25 "$LOG"; exit 4; }
grep -E "^Successfully installed|^Processing" "$LOG"

echo "=== 5. 版本证据 ==="
"$PY" -m pip list 2>/dev/null | grep -iE '^(torch|numpy|triton) ' || true
echo "site-packages 占用: $(du -sh "$VENV" | cut -f1)"

echo "=== 6. import 验证（期望 2.8.0+cpu / None / False）==="
"$PY" - <<'PYEOF'
import torch
print("torch.__version__       =", torch.__version__)
print("torch.version.cuda      =", torch.version.cuda)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.__file__          =", torch.__file__)
print("privateuse1 backend name=", torch._C._get_privateuse1_backend_name())
print("torch.get_num_threads() =", torch.get_num_threads())
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("matmul(3x4 @ 4x3) max   =", float((x @ x.T).max()))
print("torch.compile 可用       =", hasattr(torch, "compile"))
ok = (torch.__version__ == "2.8.0+cpu" and torch.version.cuda is None
      and torch.cuda.is_available() is False)
print("TORCH28_VERDICT", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 5)
PYEOF
echo "PROBE100_DONE $(date -Is)"
