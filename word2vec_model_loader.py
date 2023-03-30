import pickle
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)

with open('word2vec_model', 'w') as f:
    pickle.dump(word2vec_model, f)