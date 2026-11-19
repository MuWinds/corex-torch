set -u
# 目的：CoreX 的 nvcc 是「壳」，不打印 NVIDIA 那套 "#$ PATH= / LIBRARIES= / INCLUDES="，
#       导致 CMake 3.31 的 enable_language(CUDA) 报 "Failed to extract nvcc implicit link line"。
# 做法：在 CMake 识别编译器的那次调用（编译 CMakeCUDACompilerId.cu）以及 -v -E 探针上，
#       追加一段合法的隐式链接信息（stdout 与 stderr 各一份，避免 CMake 只收其中一路），
#       其余调用原样透传。每次调用记一行到 nvcc-calls.log，便于定位 CMake 的实际调用形态。
REAL=/usr/local/corex/bin/nvcc
LOG=$HOME/build/nvcc-calls.log

printf '%s | %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG" 2>/dev/null || true

has_e=0
has_v=0
is_probe=0
for a in "$@"; do
    [ "$a" = "-E" ] && has_e=1
    [ "$a" = "-v" ] && has_v=1
    case "$a" in
        *CMakeCUDACompilerId*) is_probe=1 ;;
        *__cmake_determine_cuda*) is_probe=1 ;;
    esac
done

if [ "$is_probe" = "1" ] || { [ "$has_e" = "1" ] && [ "$has_v" = "1" ]; }; then
    "$REAL" "$@" 2>&1
    rc=$?
    cat <<'FAKE'
#$ PATH=/usr/local/corex/bin:/usr/bin:/bin
#$ LIBRARIES=libcudart.so;libcublas.so;libcufft.so;libcurand.so;libcusolver.so;libcusparse.so;pthread;dl;rt;stdc++
#$ INCLUDES=-I/usr/local/corex/include
g++ fake.o -o /tmp/CompilerIdCUDA libcudart.so;libcublas.so;libcufft.so;libcurand.so;libcusolver.so;libcusparse.so;pthread;dl;rt;stdc++
FAKE
    cat <<'FAKE' >&2
#$ PATH=/usr/local/corex/bin:/usr/bin:/bin
#$ LIBRARIES=libcudart.so;libcublas.so;libcufft.so;libcurand.so;libcusolver.so;libcusparse.so;pthread;dl;rt;stdc++
#$ INCLUDES=-I/usr/local/corex/include
g++ fake.o -o /tmp/CompilerIdCUDA libcudart.so;libcublas.so;libcufft.so;libcurand.so;libcusolver.so;libcusparse.so;pthread;dl;rt;stdc++
FAKE
    exit $rc
fi

exec "$REAL" "$@"
