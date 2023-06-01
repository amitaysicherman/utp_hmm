import numpy as np
from hmmlearn import hmm
import pandas as pd

CODE_NAME = "code100"

with open(f"data/{CODE_NAME}.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
code = [line.split() for line in lines]

code = sum([[int(i) for i in line] for line in code], [])
n_code = max(code)+1

with open("data/phonemes.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
phonemes = [line.split() for line in lines]
phonemes = sum(phonemes, [])
unique_phonemes = np.unique(phonemes)
phonemes = [np.where(unique_phonemes == x)[0][0] for x in phonemes]

mapping = np.zeros((len(unique_phonemes), n_code))
for c, p in zip(code, phonemes):
    mapping[p, c] += 1
print(mapping.max())

for i,m in enumerate(mapping):
    mapping[i] = mapping[i]/np.sum(m)
print(mapping)
print(mapping.shape)



CODE_NAME="code100"
CLIP = -1
n_iter = 1000

# load data
phonemes_bigrams=pd.read_csv("data/bi_grams.csv",index_col=0)
trans_mat=phonemes_bigrams.values
start_line_prob=pd.read_csv("data/start_line_prob.csv",index_col=0).values.astype(float).flatten()
# load numpy arrays as type int
obs=np.load(f"data/{CODE_NAME}_one_hot.npy").argmax(axis=1).reshape(-1,1)
lens=np.loadtxt(f"data/{CODE_NAME}_lengths.txt", delimiter=",",dtype=int)

with open("data/phonemes.txt","r",encoding="utf-8") as f:
    phonemes=[line.split() for line in f.read().splitlines()]
assert len(phonemes)==len(lens)
for i,p in enumerate(phonemes):
    assert len(p)==lens[i]


# clip obs and lens:
if CLIP>0:
    lens=lens[:CLIP]
    obs=obs[:sum(lens)]
    phonemes=phonemes[:CLIP]
# print data stats
print("Data stats:")
print(f"Number of observations: {obs.shape[0]}")
print(f"Number of features: {obs.shape[1]}")
print(f"Number of states: {phonemes_bigrams.shape[0]}")
print(f"Number of sequences: {len(lens)}")



states=list([x for x in phonemes_bigrams.columns])
id2topic = dict(zip(range(len(states)), states))

model = hmm.CategoricalHMM(n_components=len(states),
        n_iter=n_iter,
        verbose=True,tol=0.5,
        init_params='',params='e')

model.n_features = obs.max()+1
model.startprob_ =start_line_prob# np.ones_like(start_line_prob)*(1/len(start_line_prob))
model.transmat_ = trans_mat#np.ones_like(trans_mat)*(1/len(trans_mat))
model.emissionprob_ = mapping
model.fit(obs, lens)
logprob, received = model.decode(obs, lens, algorithm="map")
print('logprob',logprob)

received=[id2topic[x] for x in received]

print("Accuracy: ",end="")
correct=sum([a==b for a,b in zip(sum(phonemes,[]),received)])
print(correct/len(received))