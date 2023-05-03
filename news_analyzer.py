import spacy
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

from tokenizer import word2vec_model, preprocess_and_encode
from config import news_relevance_k as K


def tokenize_article(article):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article)
    sentences = [sent.text.strip() for sent in doc.sents]
    entities = list(doc.ents)
    return sentences, entities

def average_sentence_vector(sentence):
    word_vectors = preprocess_and_encode(sentence)
    if len(word_vectors) == 0:
        return None
    avg_vector = sum(word_vectors) / len(word_vectors)
    return avg_vector.reshape(1, -1)

def find_relevant_sentences(sentences, entities, question):
    relevant_sentences = []
    question_vector = average_sentence_vector(question)
    print(question_vector)

    if question_vector is not None:
        sentence_similarities = []

        for sentence in sentences:
            sentence_vector = average_sentence_vector(sentence)
            print(sentence, sentence_vector)
            if sentence_vector is not None:
                similarity = cosine_similarity(question_vector, sentence_vector)
                sentence_similarities.append((sentence, similarity[0][0]))

        sentence_similarities.sort(key=lambda x: x[1], reverse=True)
        print(sentence_similarities)

        for i in range(K):
            # gets only the K most 'relevant' sentences to the question
            relevant_sentences.append(sentence_similarities[i][0])

    return relevant_sentences

def summarize_article(article, question):
    #word2vec_model = api.load("word2vec-google-news-300")
    sentences, entities = tokenize_article(article)
    relevant_sentences = find_relevant_sentences(sentences, entities, question)
    summary = " ".join(relevant_sentences)
    return summary

if __name__ == "__main__":
    article = """
    Tech giant XYZ Corp has announced the release of its new product, the ABC Phone. The ABC Phone is expected to hit the market on June 1st, with a starting price of $999. This innovative device features a 6.7-inch OLED display, a powerful A15 processor, and a 108MP primary camera. The company claims that the battery life of the ABC Phone can last up to two days on a single charge. In addition, the device is equipped with an advanced facial recognition system and is water-resistant up to 3 meters. XYZ Corp's CEO, John Doe, said in a press conference that the ABC Phone is a game-changer and will redefine the way people use smartphones.
    """

    question = "What is the release date and price of the ABC Phone?"
    summary = summarize_article(article, question)
    print(summary)
