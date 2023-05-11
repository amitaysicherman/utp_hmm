from hmmlearn import hmm
import numpy as np
import pandas as pd

CODE_NAME="code100"


# load data
phonemes_bigrams=pd.read_csv("data/bi_grams.csv",index_col=0)
trans_mat=pd.read_csv("data/bi_grams.csv",index_col=0).values
start_line_prob=pd.read_csv("data/start_line_prob.csv",index_col=0).values
# load numpy arrays as type int
obs=np.load(f"data/{CODE_NAME}_one_hot.npy")
lens=np.loadtxt(f"data/{CODE_NAME}_lengths.txt", delimiter=",",dtype=int)
 
# print data stats
print("Data stats:")
print(f"Number of observations: {obs.shape[0]}")
print(f"Number of features: {obs.shape[1]}")
print(f"Number of states: {phonemes_bigrams.shape[0]}")
print(f"Number of sequences: {len(lens)}")



states=list(phonemes_bigrams.columns)
id2topic = dict(zip(range(len(states)), states))
# we are more likely to talk about cats first

# For each topic, the probability of saying certain words can be modeled by
# a distribution over vocabulary associated with the categories

model = hmm.MultinomialHMM(n_components=len(states),
        # n_trials=1,
        n_iter=10,
        verbose=10,
        init_params='e',params='e')

model.n_features = obs.shape[1]
model.startprob_ = start_line_prob
model.transmat_ = trans_mat
model.fit(obs, lens)
logprob, received = model.decode(obs, lens)

print("Topics discussed:")
print([id2topic[x] for x in received])

print("Learned emission probs:")
print(model.emissionprob_)

print("Learned transition matrix:")
print(model.transmat_)
