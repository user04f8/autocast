import nltk
import numpy as np
from gensim.models import KeyedVectors

# Download the NLTK tokenizer data
nltk.download('punkt')

# Load pre-trained Word2Vec embeddings
# Replace with the actual path to the pre-trained Word2Vec model file
print('word2vec_model: getting from *.vec')
word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/nbclark/Downloads/wiki.en.vec', binary=False)
print('word2vec_model set')

def preprocess_and_encode(sentence):
    # Tokenize the input sentence
    tokens = nltk.word_tokenize(sentence.lower())
    
    # Encode each token using the Word2Vec model
    encoded_tokens = []
    for token in tokens:
        if token in word2vec_model:
            encoded_token = word2vec_model[token]
            encoded_tokens.append(encoded_token)
        else:
            print(f"Warning: '{token}' not in the vocabulary. Skipping.")
    
    return np.array(encoded_tokens)

if __name__ == '__main__':
    # Example usage
    input_sentence = "Will there be an initial public offering on either the Shanghai Stock Exchange or the Shenzhen Stock Exchange before 1 January 2016?"
    encoded_sentence = preprocess_and_encode(input_sentence)
    print(encoded_sentence)
