from gensim.models import Word2Vec
from trainw2v import load_w2vModel,load_w2vVectors
from test_utility import *
from find_similar import *
#from tsne import draw
from draw_w2v import *


def get_w2v_models(model_name_list):
    path_prefix = "F:/word2vecOutput/110epochs/0.75/"
    model_list = []
    for mode_name in model_name_list:
        model_path = path_prefix + mode_name + ".model"
        model = load_w2vModel(model_path)
        model_list.append(model)
    return model_list

if __name__=='__main__':
    # load models
    # path_prefix = "../models/50epoches/"
    # model_hs_5 = load_w2vModel(path_prefix + 'model_hs_5.model')
    # model_hs_10 = load_w2vModel(path_prefix + 'model_hs_10.model')
    # model_ns_5 = load_w2vModel(path_prefix + 'model_ns_5.model')
    # model_ns_10 = load_w2vModel(path_prefix + 'model_ns_10.model')
    model_name_list = ['model_hs_3', 'model_hs_7', 'model_ns_3', 'model_ns_7']
    #model_name_list = ['model_ns_3', 'model_ns_7']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))
    #test_is_word_in_model(zipped_model_list, list("屄  撚  仆街".split()))


    test_model_accuracy(zipped_model_list,'../data/Analogies1.txt')
    test_word_analogies(zipped_model_list)
    test_doesnt_match(zipped_model_list)
    test_similar_by_word(zipped_model_list)
    test_similarity(zipped_model_list)
    test_predict_center_word(zipped_model_list)

    #draw_random_annotation(model_list[2])
    #draw_all(model_list[2])
    #draw_similar_words(model_list[0],['國王','女王','拍拖'],5)
    #draw_similar_words2(model_list[0], ['國王', '女王', '拍拖'], 5)
    #draw_word_pairs(model_list[0],['日文','日本','俄文','俄羅斯'])
    #draw(model_hs_10)
    #draw(model_ns_5)
    #draw(model_ns_10)
