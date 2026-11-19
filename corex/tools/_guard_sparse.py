import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp'
lines=io.open(p,encoding='utf-8').read().split('\n')
targets=['void spmm(','void spgemm(','void addmv_out_sparse_csr(','void triangular_solve_out_sparse_csr(']
def find_start(sig):
    for i,l in enumerate(lines):
        if l.startswith(sig): return i
    return -1
def body_range(i):
    # find opening brace
    j=i
    while j<len(lines) and '{' not in lines[j]: j+=1
    depth=0; k=j
    while k<len(lines):
        depth+=lines[k].count('{')-lines[k].count('}')
        if depth==0 and k>j: break
        k+=1
    return j,k
inserts=[]
for sig in targets:
    i=find_start(sig)
    if i<0: print('miss',sig); continue
    j,k=body_range(i)
    if any(x[0]==j for x in inserts): print('dup',sig); continue
    inserts.append((j,k,sig))
for j,k,sig in sorted(inserts, reverse=True):
    indent='  '
    lines.insert(k, '#else\n  TORCH_CHECK(false, "CoreX: sparse op not supported: '+sig.strip()+'");\n#endif')
    lines.insert(j+1, '#if !defined(__ILUVATAR__)')
io.open(p,'w',encoding='utf-8').write('\n'.join(lines))
print('guarded', [s.strip() for _,_,s in inserts])