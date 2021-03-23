# -*- coding: utf-8 -*-
"""
This script prepares the data for the online sentence similarity survey 
done as a seminar project (Natural Language Processing WS 2020/2021) @ 
FU Berlin. We use phrases from the STS dataset: sentence pairs whose 
length (in number of words) exceeds a threshold $l$ are excluded. Here we 
set $l=25$.

@authors: Sara Bonati, Irina Kokoshko
"""

# general utility import
import numpy as np
import os
import pandas as pd
import csv
import nltk
import pickle

# directory management
wdir        = os.getcwd()

# import data (long sentences)
sentences_long   = pd.read_csv("../Data/STS.input.MSRpar.txt'),
                          sep=r'\t',
                          names=["SentenceA","SentenceB"],
                          header=None,
                          engine='python')
gs_long          = pd.read_csv("../Data/STS.gs.MSRpar.txt'),
                               header=None,
                               names=['Score'],
                               engine='python')
data_long        = pd.concat([sentences_long, gs_long], axis=1, sort=False)

# import data (short sentences)
sentences_short  = pd.read_csv("../Data/STS.input.MSRvid.txt'),
                               sep=r'\t',
                               names=["SentenceA","SentenceB"],
                               header=None,
                               engine='python')
gs_short         = pd.read_csv("../Data/STS.gs.MSRvid.txt'),
                               header=None,
                               names=['Score'],
                               engine='python')
data_short       = pd.concat([sentences_short, gs_short], axis=1, sort=False)

# merge data in 1 dataframe
data             = pd.concat([data_long, data_short],
                             axis=0,
                             sort=False,
                             ignore_index=True)

# LENGTH CHECK
# ----------------------------------------------------------------------------
# We consider max_length to be the number of words in a sentence!
max_length         = 25

sent_listA = data['SentenceA'].to_list()
sent_listB = data['SentenceB'].to_list()
sent_list  = []
for i in range(len(sent_listA)):
    if len(nltk.word_tokenize(sent_listA[i])) < max_length and len(nltk.word_tokenize(sent_listB[i])) < max_length:
        sent_list.append(sent_listA[i])
    else:
        sent_list.append(None)

sent_list  = np.array([sent_listA[s] if len(nltk.word_tokenize(sent_listA[s])) < max_length and \
                       len(nltk.word_tokenize(sent_listB[s])) < max_length else None for s in range(len(sent_listA))])
to_keep    = np.where(sent_list!=None) #sentlistA before
data       = data.iloc[to_keep[0],:]

data.head()

file_name  = 'sentences_shorter.pkl'
data.to_pickle(file_name)
