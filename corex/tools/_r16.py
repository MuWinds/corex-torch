import io, os
d='/data/home/2024240384035/build/torch24/third_party/foxi'
os.makedirs(d, exist_ok=True)
io.open(d+'/CMakeLists.txt','w').write('cmake_minimum_required(VERSION 3.5)\nproject(foxi NONE)\n')
io.open(d+'/foxi.h','w').write('#pragma once\n#define FOXI_ENABLED 0\n')
print('foxi stub ok')