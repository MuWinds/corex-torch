set -u
echo "=== REAL nvcc ==="
ls -l /usr/local/corex/bin/nvcc
echo "=== wrapper path ==="
ls -l "$HOME/build/corexroot/bin/nvcc"
echo "=== probe: -v -E -x cu ==="
"$HOME/build/corexroot/bin/nvcc" -v -E -x cu /dev/null 2>&1 | tail -25
echo "=== probe rc=$? ==="
echo "=== plain -E -x cu ==="
"$HOME/build/corexroot/bin/nvcc" -E -x cu /dev/null 2>&1 | tail -6
