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
```
install wordnet
pip install wordnet
```
```
install BERT
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer, util
```
```
install word2vector
pip install gensim
```
```python

from gensim.models.keyedvectors import KeyedVectors
model_path = 'GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

```


