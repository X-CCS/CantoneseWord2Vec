#### This script performs tests on trained fastText models ####

from gensim.models import Word2Vec
from trainw2v import load_w2vModel,load_w2vVectors
from test_utility import *
from test_function import *
from draw_w2v import *

# This function loads the models based on the model names provided
def get_w2v_models(model_name_list):
    path_prefix = "F:/word2vecOutput/optimal/"
    model_list = []
    for mode_name in model_name_list:
        model_path = path_prefix + mode_name + ".model"
        model = load_w2vModel(model_path)
        model_list.append(model)
    return model_list

if __name__=='__main__':
    # Define the names of the models you want to load from the input directory
    # Support testing multiple models simultaneously
    model_name_list = ['model_ns_7']
    model_list = get_w2v_models(model_name_list)
    zipped_model_list = list(zip(model_name_list,model_list))

    # Test models'accuracy based on the semantic analogy questions
    test_model_accuracy(zipped_model_list,'../data/Analogies1.txt')

    # Ask the model to perform semantic analogy tasks, top 5 answers returned
    test_word_analogies(zipped_model_list)

    # Test the model to find the word that doesn't belong to the group
    test_doesnt_match(zipped_model_list)

    # Test the model to find similar words of selected words
    test_similar_by_word(zipped_model_list)

    # Test the model measure the similarity between pairs of words
    test_similarity(zipped_model_list)

    # Plot 2D representations of word embeddings
    draw_word_pairs3(model_list[0], [['美國', '美金'], ['香港', '港元']])
    draw_word_pairs3(model_list[0],[['法國', '巴黎'], ['日本', '東京']])
    draw_word_pairs3(model_list[0], [['靚仔','靚女'], ['男朋友', '女朋友']])
    draw_word_pairs3(model_list[0], [['係', '是'], ['嘅', '的']])
    draw_word_pairs3(model_list[0], [['日本', '壽司'], ['美國', '薯條']])
    draw_similar_words3(model_list[0], ['屄', '意大利', '太陽', '靚仔', '香港', '飲', '自由', '薯條', '電腦', '佢', '熟石灰','萬能俠'], 10)


