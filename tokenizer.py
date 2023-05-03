import nltk
import numpy as np
import pickle
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
from math import exp

from word2vec_model_loader import WORD2VEC_MODEL_FNAME

nltk.download('punkt')

# Load pre-trained Word2Vec embeddings
print('word2vec_model: getting from pickle')
word2vec_model : KeyedVectors = pickle.load(open(WORD2VEC_MODEL_FNAME, 'rb'))
# word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)
print('word2vec_model set')

N = len(word2vec_model)

# np.hstack([word2vec_model, np.zeros((N, 3))]).ravel() 
#vec0 = [0] * len(word2vec_model[0])

def preprocess_and_encode(sentence):
    # Tokenize the input sentence
    tokens = nltk.word_tokenize(sentence.lower())

    return [word2vec_model[token] for token in tokens if token in word2vec_model]
    
    # Encode each token using the Word2Vec model
    encoded_tokens = []
    for token in tokens:
        if token in word2vec_model:
            encoded_token : np.ndarray = word2vec_model[token]
            
            encoded_tokens.append(encoded_token)
        elif False: # test with log-numeric tokens
            try: # this is so awful probably don't use exception handling for this haha but just to test for now
                n = int(token)
                encoded_token = vec0
                encoded_token[-1] = np.log(n)
                encoded_token[-2] = np.log(n) / 10
                encoded_token[-3] = np.log(n) / 100
                encoded_token[-3:-1] = np.max(1, np.min(-1, encoded_token[-3:-1]))
            except:
                print(f"Warning: '{token}' not in the vocabulary. Skipping.")
        else:
            print(f"Warning: '{token}' not in the vocabulary. Skipping.")
    
    return np.array(encoded_tokens)

    
def show_numerical_features():
    input_sentence = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    #input_sentence = "one ten hundred thousand million billion"
    #input_sentence = "zero one one two three four five seven nine twelve fifteen twenty" # roughly log
    n_words = len(input_sentence.split(' '))
    ns = [x for x in range(n_words)]
    encoded_sentence = preprocess_and_encode(input_sentence)
    print(encoded_sentence)
    n_features = len(encoded_sentence[0])
    word_avg = np.sum(encoded_sentence, axis=0) / n_words
    encoded_sentence = np.array ( [encoded_word - word_avg for encoded_word in encoded_sentence] )
    
    #r = [np.corrcoef(ns, word_feature)[0,1] for word_feature in encoded_sentence.T]
    r = [pearsonr(ns, word_feature)[1] for word_feature in encoded_sentence.T]# if abs(pearsonr(ns, word_feature)[0]) > 0.1]
    argsort_r = np.argsort(r)
    #print(r)
    #print(argsort_r)
    numerical_feature_avg = np.zeros(n_words)
    weighting_factors = np.zeros(n_features)
    weighting_bonus = 1
    print('numerical-like word features')
    print('+')
    for i in range(n_features):
        j = argsort_r[i]
        encoded_feature = encoded_sentence.T[j]
        r = pearsonr(ns, encoded_feature)[0]
        p = pearsonr(ns, encoded_feature)[1]
        if p > 0.05:
            break
        weighting_bonus -= 0.0001
        if weighting_bonus < 0:
            break
        weighting_factor = (r * weighting_bonus * (1-(20*p)))
        weighting_factors[j] = weighting_factor
        numerical_feature_avg += weighting_factor * encoded_feature
        print(f'{j} r {r:.4f} p {pearsonr(ns, encoded_feature)[1]:.4f}')
        print(encoded_feature)
        i += 1

    print('avg numerical feature (weighted sum)')
    print(numerical_feature_avg)
    print('weight vector for numerical largeness:')
    print(weighting_factors)
    #print('-')
    #for i in range(-1, -5, -1):
    #    print(argsort_r[i])
    #    print(encoded_sentence[:, argsort_r[-i]])
    

if __name__ == '__main__':
    show_numerical_features()
    
    
    # Example usage
    # input_sentence = "Will there be an initial public offering on either the Shanghai Stock Exchange or the Shenzhen Stock Exchange before 1 January 2016?"
    # encoded_sentence = preprocess_and_encode(input_sentence)
    # print(encoded_sentence)
