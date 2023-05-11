from hmmlearn import hmm
import numpy as np
import pandas as pd

CODE_NAME="code100"
CLIP = -1
n_iter = 2500

# load data
phonemes_bigrams=pd.read_csv("data/bi_grams.csv",index_col=0)
trans_mat=pd.read_csv("data/bi_grams.csv",index_col=0).values
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

for _ in range(100):
    model = hmm.CategoricalHMM(n_components=len(states),
            n_iter=n_iter,
            verbose=True,tol=0.5,
            init_params='e',params='e')

    model.n_features = obs.max()+1
    model.startprob_ = start_line_prob
    model.transmat_ = trans_mat
    model.fit(obs, lens)
    logprob, received = model.decode(obs, lens)

    print('logprob',logprob)

    received=[id2topic[x] for x in received]

    print("Accuracy: ",end="")
    correct=sum([a==b for a,b in zip(sum(phonemes,[]),received)])
    print(correct/len(received))

    np.save(f"data/{CODE_NAME}_emissionprob_{int(logprob)}_{int(100*correct/len(received))}.npy",model.emissionprob_)
