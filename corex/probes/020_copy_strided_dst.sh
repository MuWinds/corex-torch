set -u
# 目的：019 的逐阶段比对定位到「5b 切片写」是第一个发散点（max_abs 4.0，其余阶段都在 1e-5 量级）。
# 这一步在 torch 里是 `dst[..., 0::2] = src`，落到 `aten::copy_`（目的张量是**非连续视图**）。
# 这份探针只做一件事：把「拷贝」的四种方向组合拆开，看是哪一种错——
#   A 连续 dst ← 连续 src（基线）
#   B 非连续 dst ← 连续 src（切片写，就是 019 撞到的）
#   C 连续 dst ← 非连续 src
#   D 非连续 dst ← 非连续 src
# 每一格都与 CPU 逐字节比。错在哪一格，就把那一格交给对应负责人修。
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

cat > "$W/copy_dst.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch

corex_torch.register()
torch.manual_seed(7)


def dev(t):
    return t.to("corex")


def check(name, fn):
    try:
        got, ref = fn()
        same = torch.equal(got.cpu(), ref)
        d = float((got.cpu().double() - ref.double()).abs().max())
        print(f"  [{'OK ' if same else 'FAIL'}] {name:42s} byte_equal={same} max_abs={d:.3e}")
        return same
    except Exception as exc:                            # noqa: BLE001
        print(f"  [FAIL] {name:42s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        return False


SRC = torch.randn(4, 8)
SRC_S = torch.randn(4, 4)          # 给切片写的源（形状与 dst[..., 0::2] 一致）


def case_a():
    """A 连续 dst ← 连续 src。"""
    d_c = torch.zeros(4, 8)
    d_d = dev(torch.zeros(4, 8))
    d_c.copy_(SRC)
    d_d.copy_(dev(SRC))
    return d_d, d_c


def case_b():
    """B 非连续 dst ← 连续 src（`dst[..., 0::2] = src`，019 的发散点）。"""
    d_c = torch.zeros(4, 8)
    d_d = dev(torch.zeros(4, 8))
    d_c[..., 0::2] = SRC_S
    d_d[..., 0::2] = dev(SRC_S)
    return d_d, d_c


def case_b2():
    """B2 同理但用显式 copy_：`dst[..., 0::2].copy_(src)`。"""
    d_c = torch.zeros(4, 8)
    d_d = dev(torch.zeros(4, 8))
    d_c[..., 0::2].copy_(SRC_S)
    d_d[..., 0::2].copy_(dev(SRC_S))
    return d_d, d_c


def case_c():
    """C 连续 dst ← 非连续 src。"""
    s_c = torch.randn(4, 8)
    d_c = torch.zeros(4, 4)
    d_d = dev(torch.zeros(4, 4))
    d_c.copy_(s_c[..., 0::2])
    d_d.copy_(dev(s_c)[..., 0::2])
    return d_d, d_c


def case_d():
    """D 非连续 dst ← 非连续 src。"""
    s_c = torch.randn(4, 8)
    d_c = torch.zeros(4, 8)
    d_d = dev(torch.zeros(4, 8))
    d_c[..., 1::2] = s_c[..., 0::2]
    d_d[..., 1::2] = dev(s_c)[..., 0::2]
    return d_d, d_c


print("=== 拷贝四方向（全部与 CPU 逐字节比）===")
ok = [check("A 连续 dst ← 连续 src", case_a),
      check("B 非连续 dst（切片写）← 连续 src", case_b),
      check("B2 显式 dst[...,0::2].copy_(src)", case_b2),
      check("C 连续 dst ← 非连续 src", case_c),
      check("D 非连续 dst ← 非连续 src", case_d)]
print("COPY_DST_VERDICT", "PASS" if all(ok) else "PARTIAL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/copy_dst.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
