set -u
compiler=$1
shift

is_cu=0
for a in "$@"; do
    case "$a" in
        *.cu) is_cu=1; break ;;
    esac
done

if [ "$is_cu" = "1" ]; then
    CX=/usr/local/corex
    CUDA_PATH=/usr/local/corex-3.2.1
    [ -d "$CUDA_PATH" ] || CUDA_PATH=$CX
    SHIM="$(dirname "$0")/corex-shim"
    exec "$CX/bin/clang++" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path="$CUDA_PATH" -I "$SHIM" "$@"
fi

exec "$compiler" "$@"
