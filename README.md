# Building a Simple Chatbot based on Semantic Similarity using SpaCy, BERT, Wordnet, Word2Vector.

In this simple project, we building simple chatbot based on semantic similarity methods, und we make comparative between the common methods in Semantic  area.
# Step 1 — Setting Up Your Environment
In this step, you will install the spaCy, BERT, Wordnet, and Word2Vector libraries that will help your chatbot understand the user’s sentences.
```
NLTK is Natural Language Tool Kit. It is used to build python programming. It helps to work with human languages data. It gives a very easy user interface. It supports classification, steaming, tagging, etc.

pip install --user -U nltk
```

spaCy is a library for advanced Natural Language Processing in Python and Cython. It's built on the very latest research, and was designed from day one to be used in real products. spaCy comes with pretrained pipelines and currently supports tokenization and training for 60+ languages.
```
install spaCy
pip install -U spacy
python -m spacy download en_core_web_md
```
```python
import spacy
nlp = spacy.load("en_core_web_md")
```

WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus. You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more.
```
install wordnet
pip install wordnet
```
Bidirectional Encoder Representations from Transformers is a transformer-based machine learning technique for natural language processing pre-training developed by Google. 
```
install BERT
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer, util
```
Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
```
install word2vector
pip install gensim
```
```python

from gensim.models.keyedvectors import KeyedVectors
model_path = 'GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

```
# Results of Comparative Methods.
Our Question : q1="the aratificial intelliegnce play crucial role in conversational AI, "
according to the results in this project as follow:
```python

print('the response based Spacy:  ',Response_spacy(q1))
```
"the response based Spacy:   this criterion depends on the ability of a computer program to impersonate a human in a real-time written conversation with a human judge, sufficiently well that the judge is unable to distinguish reliablyon the basis of the conversational content alonebetween the program and a real human."

```python
print('the response based wordnet:  ',Response_wordnet(q1))
```
"the response based wordnet:   still, there is currently no general purpose conversational artificial intelligence, and some software developers focus on the practical aspect, information retrieval."

```python
print('the response based BERT: ',Response_BERT(q1))
```
the response based BERT:  "one pertinent field of ai research is natural language processing."
```python
print('the response based word2vec:  ',Response_word2vec(q1))
```
the response based word2vec:   "this criterion depends on the ability of a computer program to impersonate a human in a real-time written conversation with a human judge, sufficiently well that the judge is unable to distinguish reliablyon the basis of the conversational content alonebetween the program and a real human."

We found the best results based on Spacy and word2vector 


