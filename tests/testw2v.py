from gensim.models import Word2Vec
from trainw2v import load_w2vModel,load_w2vVectors
from test_utility import *
from find_similar import *
#from tsne import draw
from draw_w2v import *


def get_w2v_models(model_name_list):
    path_prefix = "F:/word2vecOutput/optimal/"
    #path_prefix = "F:/word2vecOutput/80epochs/ns_20/"
    model_list = []
    for mode_name in model_name_list:
        model_path = path_prefix + mode_name + ".model"
        model = load_w2vModel(model_path)
        model_list.append(model)
    return model_list

if __name__=='__main__':
    # model_name_list = ['model_hs_3', 'model_hs_7', 'model_ns_3', 'model_ns_7']
    #model_name_list = ['model_ns_3', 'model_ns_7']
    model_name_list = ['model_ns_7']
    # model_name_list = ['model_hs_5', 'model_ns_5']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))

    #test_model_accuracy(zipped_model_list,'../data/Analogies1.txt')
    # test_word_analogies(zipped_model_list)
    # test_doesnt_match(zipped_model_list)
    #test_similar_by_word(zipped_model_list)
    test_similarity(zipped_model_list)
    # test_predict_center_word(zipped_model_list)

    # draw_random_annotation(model_list[0],sample=100)

    # draw_word_pairs3(model_list[0],[['美國', '紐約'], ['中國', '上海']])
    # draw_word_pairs3(model_list[0], [['美國', '美金'], ['香港', '港元']])
    # draw_word_pairs3(model_list[0],[['法國', '巴黎'], ['日本', '東京']])
    # draw_word_pairs3(model_list[0], [['日本', '日文'], ['香港', '粵語']])
    # draw_word_pairs3(model_list[0], [['俄羅斯','俄文'], ['德國','德文']])
    # draw_word_pairs3(model_list[0], [['男人', '女人'], ['男朋友', '女朋友']])
    # draw_word_pairs3(model_list[0], [['男人', '女人'], ['靚仔','靚女']])
    # draw_word_pairs3(model_list[0], [['靚仔','靚女'], ['男朋友', '女朋友']])
    # draw_word_pairs3(model_list[0], [['男人', '女人'], ['國王','女王']])
    #draw_word_pairs3(model_list[0], [['係', '是'], ['嘅', '的']])
    #draw_word_pairs3(model_list[0], [['吃', '食'], ['喝', '飲']])
    #draw_word_pairs(model_list[0], [['中國', '主席'], ['美國', '總統']])
    #draw_word_pairs(model_list[0], [['吃', '食'], ['喝', '飲']])
    #draw_word_pairs3(model_list[0], [['日本', '壽司'], ['美國', '薯條']])
    #draw_similar_words3(model_list[0], ['意大利'], 10)
    #draw_similar_words3(model_list[0],['太陽'],10)
    #draw_similar_words3(model_list[0], ['仆街',], 10)
    # draw_similar_words3(model_list[0], ['電腦', ], 10)
    # draw_similar_words3(model_list[0], ['薯條'], 10)
    # draw_similar_words3(model_list[0], ['屄',], 10)
    # draw_similar_words2(model_list[0], ['屄', '意大利', '太陽', '靚仔', '香港', '飲', '自由', '薯條', '電腦', '佢', '熟石灰','萬能俠'], 10)


