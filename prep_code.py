""""
prep code
"""
import numpy as np

CODE_NAME="code100"

with open(f"data/{CODE_NAME}.txt", "r",encoding="utf-8") as f:
    lines = f.read().splitlines()
lines=[line.split() for line in lines]
lines= [[int(i) for i in line] for line in lines]
n_code = max([max(line) for line in lines])+1
n_tot=sum([len(line) for line in lines])
lengths=[len(line) for line in lines]

with open(f"data/{CODE_NAME}_lengths.txt", "w",encoding="utf-8") as f:
    f.write(",".join([str(i) for i in lengths]))

code_one_hot=np.zeros((n_tot,n_code))
for i,c in enumerate(sum(lines,[])):
    code_one_hot[i,c]=1
code_one_hot=code_one_hot.astype(int)
np.save(f"data/{CODE_NAME}_one_hot.npy",code_one_hot)
