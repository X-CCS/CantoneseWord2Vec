from trainfasttext import load_fasttext,load_fasttextVectors

#from tsne import draw
from draw_w2v import *
from test_utility import *
def get_w2v_models(model_name_list):
    path_prefix = "F:/fasttextoutput/80epochs/_0.5/"
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
    model_name_list = ['model_ns_5', 'model_ns_7']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))
    #test_is_word_in_model(zipped_model_list, "女皇", is_fasttext=True)

    test_model_accuracy(zipped_model_list,'../data/Analogies1.txt')
    test_word_analogies(zipped_model_list)
    test_doesnt_match(zipped_model_list)
    test_similar_by_word(zipped_model_list)
    test_similarity(zipped_model_list)
    #test_predict_center_word(zipped_model_list)

    #draw(model_list[1])
    #draw_all(model_list[1])
    #draw(model_hs_10)
    #draw(model_ns_5)
    #draw(model_ns_10)