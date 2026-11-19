set -u
# 目的：把厂商 torch 2.1 的 42 个头文件补丁打到 2.4 源码树上，产出「自动成功 / 冲突」两份清单。
# 用法：bash port21_to_24.sh
#   输入 ~/build/pytorch-2.4.0（上游 tarball 解包）、~/build/patches21（diff_vendor_patches.sh 产出）
#   输出 ~/build/torch24（可反复重建）与 ~/build/apply_report.txt
SRC=$HOME/build/pytorch-2.4.0
DST=$HOME/build/torch24
PATCHES=$HOME/build/patches21
LOG=$HOME/build/apply_report.txt

rm -rf "$DST"
cp -a "$SRC" "$DST"

: > "$LOG"
ok=0
fail=0
for p in $(cd "$PATCHES" && find . -name '*.patch' | sed 's#^\./##' | sort); do
    rel=${p%.patch}
    case "$rel" in
        ATen/*) tgt="$DST/aten/src/$rel" ;;
        *)      tgt="$DST/$rel" ;;
    esac
    if [ ! -f "$tgt" ]; then
        echo "NO_TARGET $rel" | tee -a "$LOG"
        fail=$((fail + 1))
        continue
    fi
    out=$(patch -l -F3 --no-backup-if-mismatch "$tgt" < "$PATCHES/$p" 2>&1)
    if [ $? -eq 0 ]; then
        ok=$((ok + 1))
        echo "OK   $rel" >> "$LOG"
    else
        fail=$((fail + 1))
        echo "FAIL $rel" >> "$LOG"
        printf '%s\n' "$out" | sed 's/^/     /' >> "$LOG"
        # 留一份冲突现场，便于逐个人工看
        mkdir -p "$HOME/build/rejects/$(dirname "$rel")"
        [ -f "$tgt.rej" ] && mv "$tgt.rej" "$HOME/build/rejects/$rel.rej"
    fi
done

echo "OK=$ok FAIL=$fail"
echo "--- 冲突文件 ---"
grep '^FAIL\|^NO_TARGET' "$LOG" | sed 's/^FAIL /  /;s/^NO_TARGET /  (no target) /'
echo "--- .rej 数量 ---"
find "$HOME/build/rejects" -name '*.rej' 2>/dev/null | wc -l
