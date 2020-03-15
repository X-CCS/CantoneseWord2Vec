from gensim.models import Word2Vec
from find_similar import find_similar_words, find_similarity
from predict import predict
from tsne import draw

# load models
model_hs_5 = Word2Vec.load('../models/model_hs_5.model')
model_hs_10 = Word2Vec.load('../models/model_hs_10.model')
model_ns_5 = Word2Vec.load('../models/model_ns_5.model')
model_ns_10 = Word2Vec.load('../models/model_ns_10.model')

def similar_words(word, topn=5):
    print('Words most similar to: "' + word + '"')
    
    print("model_hs_5:")
    find_similar_words(model_hs_5,word,topn)

    print("model_hs_10:")
    find_similar_words(model_hs_10,word,topn)

    print("model_ns_5:")
    find_similar_words(model_ns_5,word,topn)

    print("model_ns_10:")
    find_similar_words(model_ns_10,word,topn)

def similarity(word1, word2):
    print("Similarity between", word1, "and", word2)
    
    print("model_hs_5:")
    find_similarity(model_hs_5, word1, word2)
    
    print("model_hs_10:")
    find_similarity(model_hs_10, word1, word2)
    
    print("model_ns_5:")
    find_similarity(model_ns_5, word1, word2)
    
    print("model_ns_10:")
    find_similarity(model_ns_10, word1, word2)

def predict_center_word(context, topn=5):
    print("Probability distribution of the center word given context words:", context)

    print("model_hs_5:")
    predict(model_hs_5, context, topn)

    print("model_hs_10:")
    predict(model_hs_10, context, topn)
    
    print("model_ns_5:")
    predict(model_ns_5, context, topn)
    
    print("model_ns_10:")
    predict(model_ns_10, context, topn)



similar_words("細路")
#similar_words("貓")
#similar_words("你嘅")

#similarity("紅", "綠")
#similarity("你", "我")

#predict_center_word(["我", "蘋果"])
#predict_center_word(["飛機", "引擎"])

draw(model_hs_5)
#draw(model_hs_10)
#draw(model_ns_5)
#draw(model_ns_10)
