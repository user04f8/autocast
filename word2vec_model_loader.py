import pickle
from gensim.models import KeyedVectors

with open('word2vec_model', 'wb') as f: # 'wb' == binary mode
    word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)
  
    pickle.dump(word2vec_model, f)




