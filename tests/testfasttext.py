#### This script performs tests on trained fastText models ####

from trainfasttext import load_fasttext,load_fasttextVectors

from draw_w2v import *
from test_utility import *

# This function loads the models based on the model names provided
def get_fasttext_models(model_name_list):

    # Define the path to find trained model files
    path_prefix = "F:/fasttextoutput/Optimal/"
    model_list = []
    for mode_name in model_name_list:
        model_path = path_prefix + mode_name + ".model"
        model = load_fasttext(model_path)
        model_list.append(model)
    return model_list

if __name__=='__main__':
    # Define the names of the models you want to load from the input directory
    # Support testing multiple models simultaneously
    model_name_list = ['model_hs_5']
    model_list = get_fasttext_models(model_name_list)
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

    # Test the model to find similar words of a OOV word
    test_is_word_in_model(zipped_model_list, list("叉燒飯".split()), is_fasttext=True)
    find_similar_words(model_list[0], "叉燒飯", 5)
    test_is_word_in_model(zipped_model_list, list("蓮蓉月餅".split()), is_fasttext=True)
    find_similar_words(model_list[0], "蓮蓉月餅", 5)

    # Plot 2D representations of word embeddings
    draw_word_pairs3(model_list[0],[['法國', '巴黎'], ['日本', '東京']])
    draw_word_pairs3(model_list[0], [['俄羅斯','俄文'], ['德國','德文']])
    draw_word_pairs3(model_list[0], [['靚仔','靚女'], ['男朋友', '女朋友']])
    draw_word_pairs3(model_list[0], [['係', '是'], ['嘅', '的']])
    draw_word_pairs3(model_list[0], [['美國', '美金'], ['香港', '港紙']])
    draw_similar_words3(model_list[0], ['屄', '巴西', '太陽', '靚仔', '香港', '飲', '自由', '薯條', '電腦', '佢', '多瑙河','風濕'], 10)
