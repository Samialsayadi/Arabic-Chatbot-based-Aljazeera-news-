# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import numpy as np
from pyemd import emd
import nltk

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None

def tagged_to_synsets(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return []
    return wn.synsets(word, wn_tag)

def get_counts(sentence, vocab):
    weights = np.zeros(len(vocab))
    for w in sentence:
        if w not in vocab:
            continue
        weights[vocab.index(w)] += 1
    return weights / sum(weights)

def sim3(sentence1, sentence2):
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    vocab = [pair for pair in sorted(set(sentence1).union(set(sentence2))) if penn_to_wn(pair[1])]

    w1 = get_counts(sentence1, vocab)
    w2 = get_counts(sentence2, vocab)

    synsets = [tagged_to_synsets(*tagged_word) for tagged_word in vocab]

    similarities = np.array([[
        max([s1.path_similarity(s2) or 0 for s1 in w1 for s2 in w2], default=0)
        for w2 in synsets] for w1 in synsets]
    )
    distances = np.sqrt(2*(1-similarities))
    distance = emd(w1, w2, distances)
    similarity = 1 - distance**2 / 2
    return similarity
# -


