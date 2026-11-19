import io, os
d='/data/home/2024240384035/build/torch24/third_party/foxi-src'
io.open(d+'/dummy.cc','w').write('// CoreX stub\n')
io.open(d+'/CMakeLists.txt','w',encoding='utf-8').write('''cmake_minimum_required(VERSION 3.18)
project(foxi CXX)
add_library(foxi_loader STATIC dummy.cc)
''')
print('foxi stub lib ready')