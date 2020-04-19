from gensim.models import Word2Vec
from gensim.models import keyedvectors

def find_similar_words(model, target, topn=5):
    for word in model.wv.similar_by_word(target,topn=topn):
        print(word[0], word[1])
    print('-----------------------------------------------------')


def find_similarity(model, word1, word2):
    print(model.wv.similarity(word1, word2))
    print(model.wv.distance(word1,word2))
    print('-----------------------------------------------------')


def find_answer_analogy_question(model,positive,negative,topn=5):
    print(model.wv.most_similar(positive=positive, negative=negative,topn=topn))
    print('-----------------------------------------------------')


def find_doesnt_match(model,words):
    print(model.wv.doesnt_match(words))
    print('-----------------------------------------------------')