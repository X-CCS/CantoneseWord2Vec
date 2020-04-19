from gensim.models import Word2Vec
from find_similar import find_similar_words, find_similarity
from predict import predict
from tsne import draw

# load models
model_hs_5 = Word2Vec.load('../models/model_hs_5.model')
#model_hs_10 = Word2Vec.load('../models/model_hs_10.model')
model_ns_5 = Word2Vec.load('../models/model_ns_5.model')
#model_ns_10 = Word2Vec.load('../models/model_ns_10.model')

def similar_words(word, topn=5):
    print('Words most similar to: "' + word + '"')
    
    print("model_hs_5:")
    find_similar_words(model_hs_5,word,topn)

#    print("model_hs_10:")
#    find_similar_words(model_hs_10,word,topn)

    print("model_ns_5:")
    find_similar_words(model_ns_5,word,topn)

#    print("model_ns_10:")
#    find_similar_words(model_ns_10,word,topn)

def similarity(word1, word2):
    print("Similarity between", word1, "and", word2)
    
    print("model_hs_5:")
    find_similarity(model_hs_5, word1, word2)
    
#    print("model_hs_10:")
#    find_similarity(model_hs_10, word1, word2)

    print("model_ns_5:")
    find_similarity(model_ns_5, word1, word2)
    
#    print("model_ns_10:")
#    find_similarity(model_ns_10, word1, word2)

def predict_center_word(context, topn=5):
    print("Probability distribution of the center word given context words:", context)

#    print("model_hs_5:")
#    predict(model_hs_5, context, topn)

#    print("model_hs_10:")
#    predict(model_hs_10, context, topn)

    print("model_ns_5:")
    predict(model_ns_5, context, topn)
    
#    print("model_ns_10:")
#    predict(model_ns_10, context, topn)



similar_words("太陽",3)
similar_words("貓", 3)
similar_words("屌", 3)
similar_words("紅", 3)
similar_words("蘋果", 3)
similar_words("中國", 3)

#similarity("紅", "綠")
#similarity("你", "我")
#
predict_center_word(["我", "西瓜"])
predict_center_word(["飛機", "引擎"])

#draw(model_hs_5)
#draw(model_hs_10)
#draw(model_ns_5)
#draw(model_ns_10)



#print('most_similar:')
#print('------------hs--------------')
#print(model_hs_5.wv.most_similar(positive=['壽司','美國'],negative=['日本']))
#print()
#print(model_hs_5.wv.most_similar(positive=['日文','俄羅斯'],negative=['日本']))
#print()
#print(model_hs_5.wv.most_similar(positive=['日文','德國'],negative=['日本']))
#print()
#print(model_hs_5.wv.most_similar(positive=['月亮','男'],negative=['太陽']))
#print()
#print()
#print('------------ns--------------')
#print(model_ns_5.wv.most_similar(positive=['壽司','美國'],negative=['日本']))
#print()
#print(model_ns_5.wv.most_similar(positive=['日文','俄羅斯'],negative=['日本']))
#print()
#print(model_ns_5.wv.most_similar(positive=['日文','德國'],negative=['日本']))
#print()
#print(model_ns_5.wv.most_similar(positive=['月亮','男'],negative=['太陽']))
#print()
#
#
#
#
#print('doesnt match:')
#print(model_hs_5.wv.doesnt_match("紅 綠 藍 火".split()))
#print(model_hs_5.wv.doesnt_match("星星 月亮 太陽 石頭".split()))
#print(model_hs_5.wv.doesnt_match("魚 蝙蝠 蛇 梨".split()))
#print(model_hs_5.wv.doesnt_match("俄羅斯 美國 印度 導彈".split()))
#print(model_hs_5.wv.doesnt_match("食 飲 吞 跳".split()))
