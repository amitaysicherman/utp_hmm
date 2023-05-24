"""

"""
import numpy as np
import pandas as pd


with open("data/phonemes.txt", "r",encoding="utf-8") as f:
    lines = f.read().splitlines()
lines=[line.split() for line in lines]
unique_phonemes=np.unique(sum(lines,[]))

bi_grams=np.zeros((len(unique_phonemes),len(unique_phonemes)))
for line in lines:
    for i in range(len(line)-1):
        bi_grams[np.where(unique_phonemes==line[i]),np.where(unique_phonemes==line[i+1])]+=1
for i in range(len(unique_phonemes)):
    bi_grams[i,:] = bi_grams[i,:] / sum(bi_grams[i,:])




start_line_prob=np.zeros(len(unique_phonemes))
for line in lines:
    start_line_prob[np.where(unique_phonemes==line[0])]+=1
start_line_prob=start_line_prob/np.sum(start_line_prob)
pd.DataFrame(start_line_prob,index=unique_phonemes).to_csv("data/start_line_prob.csv")

# calsulate the fequency of each phoneme:
freq=np.zeros(len(unique_phonemes))
for line in lines:
    for i in range(len(line)):
        freq[np.where(unique_phonemes==line[i])]+=1
freq=freq/np.sum(freq)
pd.DataFrame(freq,index=unique_phonemes).to_csv("data/freq.csv")



pd.DataFrame(bi_grams,index=unique_phonemes,columns=unique_phonemes).to_csv("data/bi_grams.csv")







