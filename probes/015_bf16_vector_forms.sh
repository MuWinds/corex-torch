set -u
# 目的：bf16 的**向量**转换到底该发什么。014 已经定位到唯一一条编不过的指令：
#     %n = fptrunc <8 x float> %x to <8 x bfloat>
# 标量形态（fpext/fptrunc → llvm.bi.bfloat2float / llvm.bi.float2bfloat）已经验证可编译（012），
# 现在要确定向量形态走哪条路。四个候选，逐个喂给 llc 看谁过：
#   V1 标量 intrinsic（对照组，已知能过）
#   V2 向量 intrinsic：call <8 x bfloat> @llvm.bi.float2bfloat(<8 x float>) —— 若 llc 认这个重载，
#      改写只需把 fptrunc 换成 call，最省事、语义与厂商一致
#   V3 整数位运算·截断：bitcast → lshr（显式向量常量，不用 splat）→ trunc → bitcast
#   V4 整数位运算·RNE：在 V3 上补 +0x7FFF 与最低位
# 另外单测一条向量展宽 fpext <8 x bfloat> → <8 x float> 的两种写法。
# 这份探针不需要卡（只调 llc），但仍然走沙箱队列，因为登录节点没有 /usr/local/corex。
CX=/usr/local/corex
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
mkdir -p "$W/bfv"
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 取一份真实 biir 的头部（datalayout/triple 必须与真编译一致）==="
if [ -f "$W/mm_biir.ll" ]; then
    head -3 "$W/mm_biir.ll" > "$W/bfv/head.ll"
else
    printf 'target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"\ntarget triple = "bi-iluvatar-ilurt"\n' > "$W/bfv/head.ll"
fi
cat "$W/bfv/head.ll"

run_llc() {   # $1=名字 $2=文件
    if $CX/bin/llc -march=bi -mcpu=ivcore10 -mtriple=bi-iluvatar-ilurt -filetype=obj \
            "$2" -o "${2%.ll}.o" 2>"${2%.ll}.err"; then
        echo "  $1: llc OK ($(stat -c%s "${2%.ll}.o" 2>/dev/null) B)"
    else
        echo "  $1: llc FAIL -> $(grep -m1 -E 'error|Cannot select' "${2%.ll}.err" | cut -c1-140)"
    fi
}

echo "=== 1. V1 标量 intrinsic（对照）==="
cat "$W/bfv/head.ll" > "$W/bfv/v1.ll"
cat >> "$W/bfv/v1.ll" <<'EOF'
declare bfloat @llvm.bi.float2bfloat(float)
declare float @llvm.bi.bfloat2float(bfloat)
define void @k1(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %f = load float, ptr addrspace(1) %in, align 4
  %b = call bfloat @llvm.bi.float2bfloat(float %f)
  %g = call float @llvm.bi.bfloat2float(bfloat %b)
  store float %g, ptr addrspace(1) %out, align 4
  ret void
}
EOF
run_llc V1 "$W/bfv/v1.ll"

echo "=== 2. V2 向量 intrinsic（显式重载声明）==="
cat "$W/bfv/head.ll" > "$W/bfv/v2.ll"
cat >> "$W/bfv/v2.ll" <<'EOF'
declare <8 x bfloat> @llvm.bi.float2bfloat(<8 x float>)
define void @k2(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load <8 x float>, ptr addrspace(1) %in, align 32
  %b = call <8 x bfloat> @llvm.bi.float2bfloat(<8 x float> %v)
  store <8 x bfloat> %b, ptr addrspace(1) %out, align 16
  ret void
}
EOF
run_llc V2 "$W/bfv/v2.ll"

echo "=== 3. V3 整数位运算·截断（向量）==="
cat "$W/bfv/head.ll" > "$W/bfv/v3.ll"
cat >> "$W/bfv/v3.ll" <<'EOF'
define void @k3(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load <8 x float>, ptr addrspace(1) %in, align 32
  %i = bitcast <8 x float> %v to <8 x i32>
  %s = lshr <8 x i32> %i, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %t = trunc <8 x i32> %s to <8 x i16>
  %b = bitcast <8 x i16> %t to <8 x bfloat>
  store <8 x bfloat> %b, ptr addrspace(1) %out, align 16
  ret void
}
EOF
run_llc V3 "$W/bfv/v3.ll"

echo "=== 4. V4 整数位运算·RNE（向量）==="
cat "$W/bfv/head.ll" > "$W/bfv/v4.ll"
cat >> "$W/bfv/v4.ll" <<'EOF'
define void @k4(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load <8 x float>, ptr addrspace(1) %in, align 32
  %i = bitcast <8 x float> %v to <8 x i32>
  %lo = lshr <8 x i32> %i, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %lsb = and <8 x i32> %lo, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %c1 = add <8 x i32> %i, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %c2 = add <8 x i32> %c1, %lsb
  %hi = lshr <8 x i32> %c2, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %t = trunc <8 x i32> %hi to <8 x i16>
  %b = bitcast <8 x i16> %t to <8 x bfloat>
  store <8 x bfloat> %b, ptr addrspace(1) %out, align 16
  ret void
}
EOF
run_llc V4 "$W/bfv/v4.ll"

echo "=== 5. 向量展宽两条写法 ==="
cat "$W/bfv/head.ll" > "$W/bfv/w1.ll"
cat >> "$W/bfv/w1.ll" <<'EOF'
declare <8 x float> @llvm.bi.bfloat2float(<8 x bfloat>)
define void @kw1(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %b = load <8 x bfloat>, ptr addrspace(1) %in, align 16
  %f = call <8 x float> @llvm.bi.bfloat2float(<8 x bfloat> %b)
  store <8 x float> %f, ptr addrspace(1) %out, align 32
  ret void
}
EOF
run_llc W1-向量intrinsic "$W/bfv/w1.ll"

cat "$W/bfv/head.ll" > "$W/bfv/w2.ll"
cat >> "$W/bfv/w2.ll" <<'EOF'
define void @kw2(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %b = load <8 x bfloat>, ptr addrspace(1) %in, align 16
  %i = bitcast <8 x bfloat> %b to <8 x i16>
  %z = zext <8 x i16> %i to <8 x i32>
  %s = shl <8 x i32> %z, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %f = bitcast <8 x i32> %s to <8 x float>
  store <8 x float> %f, ptr addrspace(1) %out, align 32
  ret void
}
EOF
run_llc W2-位运算展宽 "$W/bfv/w2.ll"

echo "=== 6. 结论 ==="
for f in v1 v2 v3 v4 w1 w2; do
    if [ -f "$W/bfv/$f.o" ]; then echo "  $f: PASS"; else echo "  $f: FAIL"; fi
done
echo "BFV_DONE"
