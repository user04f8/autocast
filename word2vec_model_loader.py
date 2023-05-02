WORD2VEC_MODEL_FNAME = 'word2vec_model'
if __name__ == '__main__':
    import pickle
    from gensim.models import KeyedVectors

    
    with open(WORD2VEC_MODEL_FNAME, 'wb') as f: # 'wb' == binary mode
        word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)
    
        pickle.dump(word2vec_model, f)




