set -eu
# 目的：把厂商 torch-2.1.0+corex.3.2.1 wheel 里带 __ILUVATAR 的头文件，与上游 v2.1.0 源码逐文件 diff，
#       产出「厂商到底改了哪几行」的补丁集，作为 torch 2.4 移植的输入。
# 用法：bash diff_vendor_patches.sh [上游源码根] [输出目录]
#       默认 ~/build/pytorch-2.1.0 → ~/build/patches21
P=/data/software/Anaconda3/2024.10-1/envs/py310/lib/python3.10/site-packages/torch/include
U=${1:-$HOME/build/pytorch-2.1.0}
OUT=${2:-$HOME/build/patches21}

rm -rf "$OUT"; mkdir -p "$OUT"
cd "$P"
miss=0
for f in $(grep -rl '__ILUVATAR' . | sed 's#^\./##'); do
    case "$f" in
        ATen/*) src="$U/aten/src/$f" ;;
        *)      src="$U/$f" ;;
    esac
    if [ -f "$src" ]; then
        mkdir -p "$OUT/$(dirname "$f")"
        diff -u "$src" "$P/$f" > "$OUT/$f.patch" || true
    else
        echo "MISSING $f -> $src"
        miss=$((miss + 1))
    fi
done

echo "PATCH_FILES=$(find "$OUT" -name '*.patch' | wc -l) MISSING=$miss"
echo "--- 改动行数（前 50）---"
for p in $(find "$OUT" -name '*.patch'); do
    n=$(grep -c '^[+-][^+-]' "$p" || true)
    printf '%6s %s\n' "${n:-0}" "${p#$OUT/}"
done | sort -rn | head -50
