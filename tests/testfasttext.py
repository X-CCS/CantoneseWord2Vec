from trainfasttext import load_fasttext,load_fasttextVectors

#from tsne import draw
from draw_w2v import draw_all, draw
from test_utility import *
def get_w2v_models(model_name_list):
    path_prefix = "F:/fasttextoutput/20epochs/0.75/"
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
    model_name_list = ['model_hs_3', 'model_hs_7', 'model_ns_3', 'model_ns_7']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))

    #(zipped_model_list, "女皇", is_fasttext=True)
    #test_word_analogies(zipped_model_list)
    #test_doesnt_match(zipped_model_list)
    #test_similar_by_word(zipped_model_list)
    #test_similarity(zipped_model_list)

    #draw(model_list[1])
    #draw_all(model_list[1])
    #draw(model_hs_10)
    #draw(model_ns_5)
    #draw(model_ns_10)