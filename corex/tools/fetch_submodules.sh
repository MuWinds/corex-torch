set -eu
# 目的：v2.4.0 的 tarball 不含子模块，按 GitHub API 返回的锁定 sha 逐个抓归档并解到 third_party/<name>。
# 用法：bash fetch_submodules.sh [源码树，默认 ~/build/torch24]
T=${1:-$HOME/build/torch24}
cd "$T"

curl -s "https://api.github.com/repos/pytorch/pytorch/contents/third_party?ref=v2.4.0" -o /tmp/tp24.json

fetch() {
    name=$1
    repo=$2
    if [ -d "third_party/$name" ] && [ -n "$(ls -A "third_party/$name" 2>/dev/null)" ]; then
        echo "$name SKIP(alreadypresent)"
        return 0
    fi
    sha=$(python3 - "$name" <<'PY'
import json, sys
d = json.load(open('/tmp/tp24.json'))
for e in d:
    if e.get('name') == sys.argv[1]:
        print(e['sha'])
        break
PY
)
    if [ -z "$sha" ]; then echo "NO_SHA $name"; return 1; fi
    mkdir -p "third_party/$name"
    curl -sSL "https://codeload.github.com/$repo/tar.gz/$sha" | tar xz -C "third_party/$name" --strip-components=1
    echo "$name $sha OK"
}

fetch fmt        fmtlib/fmt
fetch cutlass    NVIDIA/cutlass
fetch pocketfft  mreineck/pocketfft
fetch nlohmann   nlohmann/json
fetch flatbuffers google/flatbuffers
# CPU 侧必需（PyTorch 2.4 的 tarball 不含这些子模块内容）
fetch pybind11   pybind/pybind11
fetch cpuinfo    pytorch/cpuinfo
fetch eigen      eigen-mirror/eigen
fetch FP16       Maratyszcza/FP16
fetch FXdiv      Maratyszcza/FXdiv
fetch pthreadpool Maratyszcza/pthreadpool
fetch sleef      shibatch/sleef
fetch protobuf   protocolbuffers/protobuf
fetch python-peachpy  peachpy/peachpy

echo "--- 结果 ---"
for d in fmt cutlass pocketfft nlohmann flatbuffers pybind11 cpuinfo eigen FP16 FXdiv pthreadpool sleef protobuf python-peachpy; do
    printf '%-16s %s\n' "$d" "$(ls third_party/$d | wc -l) 项"
done
