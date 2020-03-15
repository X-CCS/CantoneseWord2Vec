from gensim.models import Word2Vec

def find_similar_words(model, target, topn=5):
    for word in model.wv.similar_by_word(target,topn=topn):
        print(word[0], word[1])
    print('-----------------------------------------------------')

def find_similarity(model, word1, word2):
    print(model.similarity(word1, word2))
    print('-----------------------------------------------------')
