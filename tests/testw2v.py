from gensim.models import Word2Vec
from trainw2v import load_w2vModel,load_w2vVectors
from find_similar import find_similar_words, find_similarity, find_answer_analogy_question, find_doesnt_match
from predict import predict
from tsne import draw

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

def get_anwer_to_analogy_question(positive,negative):
    print(positive[0] + " is to " + negative[0] + "as "+ positive[1] + "is to?\n")

    print("model_hs_5:")
    print(find_answer_analogy_question(model_hs_5,positive,negative))

    print("model_hs_10:")
    print(find_answer_analogy_question(model_hs_10, positive, negative))

    print("model_ns_5:")
    print(find_answer_analogy_question(model_ns_5, positive, negative))

    print("model_ns_10:")
    print(find_answer_analogy_question(model_ns_10, positive, negative))

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

def get_model_trainingloss():
    print("Training loss of each model:\n")

    print("model_hs_5:%f" % model_hs_5.get_latest_training_loss())

    print("model_hs_10:%f" % model_hs_10.get_latest_training_loss())

    print("model_ns_5:%f" % model_ns_5.get_latest_training_loss())

    print("model_ns_10:%f" % model_ns_10.get_latest_training_loss())

if __name__=='__main__':
    # load models
    path_prefix = "../models/50epoches/"
    model_hs_5 = load_w2vModel(path_prefix + 'model_hs_5.model')
    model_hs_10 = load_w2vModel(path_prefix + 'model_hs_10.model')
    model_ns_5 = load_w2vModel(path_prefix + 'model_ns_5.model')
    model_ns_10 = load_w2vModel(path_prefix + 'model_ns_10.model')

    get_model_trainingloss()
    #similar_words("太陽",3)
    #similar_words("貓", 3)
    #similar_words("屌", 3)

    #similarity("紅", "綠")
    #similarity("你", "我")

    #predict_center_word(["我", "蘋果"])
    #predict_center_word(["飛機", "引擎"])

    #draw(model_hs_5)
    #draw(model_hs_10)
    #draw(model_ns_5)
    #draw(model_ns_10)