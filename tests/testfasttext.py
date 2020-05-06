from trainfasttext import load_fasttext,load_fasttextVectors

#from tsne import draw
from draw_w2v import *
from test_utility import *
def get_w2v_models(model_name_list):
    path_prefix = "F:/fasttextoutput/Optimal/"
    #path_prefix = "F:/fasttextoutput/80epochs/ns_20/"
    model_list = []
    for mode_name in model_name_list:
        model_path = path_prefix + mode_name + ".model"
        model = load_fasttext(model_path)
        model_list.append(model)
    return model_list

if __name__=='__main__':
    # load models
    # path_prefix = "../models/50epoches/"
    # model_hs_5 = load_w2vModel(path_prefix + 'model_hs_5.model')
    # model_hs_10 = load_w2vModel(path_prefix + 'model_hs_10.model')
    # model_ns_5 = load_w2vModel(path_prefix + 'model_ns_5.model')
    # model_ns_10 = load_w2vModel(path_prefix + 'model_ns_10.model')
    #model_name_list = ['model_hs_5', 'model_hs_7', 'model_ns_5', 'model_ns_7']
    #model_name_list = ['model_ns_5', 'model_ns_7']
    model_name_list = ['model_hs_5']
    # model_name_list = ['model_hs_7', 'model_ns_7']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))
    # print(len(model_list[0].wv['女皇']))
    #test_model_accuracy(zipped_model_list,'../data/Analogies1.txt')
    # test_word_analogies(zipped_model_list)
    # test_doesnt_match(zipped_model_list)
    # test_similar_by_word(zipped_model_list)
    test_similarity(zipped_model_list)
    #test_similar_by_word_addition(zipped_model_list)
    #draw_random_annotation(model_list[0],sample=150)
    # draw_word_pairs3(model_list[0],[['美國', '紐約'], ['中國', '上海']])
    # draw_word_pairs3(model_list[0],[['法國', '巴黎'], ['日本', '東京']])
    # draw_word_pairs3(model_list[0], [['日本', '日文'], ['香港', '粵語']])
    # draw_word_pairs3(model_list[0], [['俄羅斯','俄文'], ['德國','德文']])
    # draw_word_pairs3(model_list[0], [['靚仔','靚女'], ['男朋友', '女朋友']])
    # draw_word_pairs3(model_list[0], [['男人', '女人'], ['國王','女王']])
    # draw_word_pairs3(model_list[0], [['係', '是'], ['嘅', '的']])
    # draw_word_pairs3(model_list[0], [['吃', '食'], ['喝', '飲']])
    # draw_word_pairs3(model_list[0], [['中國', '主席'], ['美國', '總統']])
    # draw_word_pairs3(model_list[0], [['美國', '美金'], ['香港', '港紙']])
    # draw_similar_words3(model_list[0],['太陽'],10)
    # draw_similar_words3(model_list[0], ['薯條'], 10)
    # draw_similar_words3(model_list[0], ['屄',], 10)
    # draw_similar_words3(model_list[0], ['電腦', ], 10)
    # draw_similar_words3(model_list[0], ['巴西', ], 10)
    # draw_similar_words3(model_list[0], ['屄', '巴西', '太陽', '靚仔', '香港', '飲', '自由', '薯條', '電腦', '佢', '多瑙河','風濕'], 10)
    # test_is_word_in_model(zipped_model_list, list("男藝人".split()), is_fasttext=True)
    # find_similar_words(model_list[0],"男藝人",5)
    # test_is_word_in_model(zipped_model_list, list("叉燒飯".split()), is_fasttext=True)
    # find_similar_words(model_list[0], "叉燒飯", 5)
    # test_is_word_in_model(zipped_model_list, list("建制派".split()), is_fasttext=True)
    # find_similar_words(model_list[0], "建制派", 5)
    # test_is_word_in_model(zipped_model_list, list("蓮蓉月餅".split()), is_fasttext=True)
    # find_similar_words(model_list[0], "蓮蓉月餅", 5)