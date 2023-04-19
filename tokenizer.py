import nltk
import numpy as np
import pickle
from gensim.models import KeyedVectors

WORD2VEC_MODEL_FNAME = 'word2vec_model'

# Download the NLTK tokenizer data
nltk.download('punkt')

# Load pre-trained Word2Vec embeddings
# Replace with the actual path to the pre-trained Word2Vec model file
print('word2vec_model: getting from pickle')
word2vec_model : KeyedVectors = pickle.load(open(WORD2VEC_MODEL_FNAME, 'rb'))
# word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)
print('word2vec_model set')

N = len(word2vec_model)

# np.hstack([word2vec_model, np.zeros((N, 3))]).ravel() # DOESN'T WORK due to it being a KeyedVector not ndarray
# last three zeroes reserved for floats/integers


vec0 = [0] * len(word2vec_model[0])

def preprocess_and_encode(sentence):
    # Tokenize the input sentence
    tokens = nltk.word_tokenize(sentence.lower())
    
    # Encode each token using the Word2Vec model
    encoded_tokens = []
    for token in tokens:
        if token in word2vec_model:
            encoded_token : np.ndarray = word2vec_model[token]
            
            encoded_tokens.append(encoded_token)
        elif True:
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

if __name__ == '__main__':
    show_numerical_like()
    
    
    # Example usage
    # input_sentence = "Will there be an initial public offering on either the Shanghai Stock Exchange or the Shenzhen Stock Exchange before 1 January 2016?"
    # encoded_sentence = preprocess_and_encode(input_sentence)
    # print(encoded_sentence)
    
def show_numerical_like():
    input_sentence = "zero one two three four five six seven eight nine ten eleven twelve thirteen"
    ns = [x for x in range(14)]
    encoded_sentence = preprocess_and_encode(input_sentence)
    print(encoded_sentence)
    #word_avg = np.sum(encoded_sentence, axis=0) / 14
    #print(word_avg)
    r = [np.corrcoef(ns, word_feature)[0,1] for word_feature in encoded_sentence.T]
    argsort_r = np.argsort(r)
    print(r)
    print(argsort_r)
    print('numerical-like word features')
    print('-')
    print(encoded_sentence[:, argsort_r[0]])
    print('+')
    print(encoded_sentence[:, argsort_r[-1]])
    