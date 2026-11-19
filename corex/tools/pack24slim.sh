#!/bin/bash
cd $HOME/build
rm -f torch24-src.tgz torch24-src.tar.zst
EX=()
for d in build dist .git torch/lib torch/bin torch/lib64; do EX+=(--exclude="torch24/$d"); done
for d in benchmark cpp-httplib cpuinfo cudnn_frontend cutlass eigen fbgemm flatbuffers fmt gloo googletest ideep ittapi mkl_headers NNPACK nlohmann onnx onnx-1.16.1 opentelemetry-cpp pocketfft protobuf psimd pybind11 python-peachpy sleef tensorpipe valgrind-headers VulkanMemoryAllocator XNNPACK; do EX+=(--exclude="torch24/third_party/$d"); done
EX+=(--exclude='*.so' --exclude='*.a' --exclude='*.o' --exclude='*.pyc' --exclude='compile_commands.json' --exclude='__pycache__')
tar -cf - "${EX[@]}" torch24 2>/dev/null | zstd -19 -T0 -q -o torch24-src.tar.zst
echo tar_pipe_rc=$?
ls -la torch24-src.tar.zst
echo "uncompressed=$(du -sh --exclude=build --exclude=dist torch24 | cut -f1)"
