import io
p='/data/home/2024240384035/build/torch24/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
a='add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx EXCLUDE_FROM_ALL)'
b='add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx-1.16.1 EXCLUDE_FROM_ALL)'
if a in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(a,b)); print('path patched')
else:
    print('ok' if b in s else 'MISS')