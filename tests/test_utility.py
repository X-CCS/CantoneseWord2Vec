from find_similar import *
from predict import *


def similar_words(zipped_model_list, word, topn=5):
    print('Words most similar to: "' + word + '"')
    for model_name, model in zipped_model_list:
        print(model_name)
        find_similar_words(model,word,topn)
    print("***************************************************")


def similarity(zipped_model_list, word1, word2):
    print("Similarity between", word1, "and", word2)
    for model_name, model in zipped_model_list:
        print(model_name)
        find_similarity(model, word1, word2)
    print("***************************************************")


def get_anwer_to_analogy_question(zipped_model_list, positive, negative):
    print(negative[0] + " is to " + positive[0] + " as "+ positive[1] + " is to?\n")
    for model_name, model in zipped_model_list:
        print(model_name)
        find_answer_analogy_question(model, positive, negative)
    print("***************************************************")


def get_word_doesnt_match(zipped_model_list, words):
    print("The word that doesn't belongs to {} is?".format(words))
    for model_name, model in zipped_model_list:
        print(model_name)
        find_doesnt_match(model, words)
    print("***************************************************")


def predict_center_word(zipped_model_list, context, topn=5):
    print("Probability distribution of the center word given context words:", context)
    for model_name, model in zipped_model_list:
        print(model_name)
        predict(model, context, topn)
    print("***************************************************")


def get_model_trainingloss(zipped_model_list):
    print("Training loss of each model:\n")
    for model_name, model in zipped_model_list:
        print(model_name + ":%f" % model.get_latest_training_loss())
    print("***************************************************")


def test_model_accuracy(zipped_model_list,path_to_file):
    print("The accuracy of each model in terms of word analogies ")
    for model_name,model in zipped_model_list:
        print(model_name)
        get_accuracy(model,path_to_file)
    print("***************************************************")


def test_model_evaluation_score(zipped_model_list,path_to_file):
    print("The accuracy of each model in terms of word ")
    for model_name,model in zipped_model_list:
        print(model_name)
        get_evaluation_score(model,path_to_file)
    print("***************************************************")


def test_word_analogies(zipped_model_list):
    get_anwer_to_analogy_question(zipped_model_list, positive=['國王', '女'], negative=['男'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['女人', '國王'], negative=['女王'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['女王', '男人'], negative=['女人'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['皇后', '國王'], negative=['女王'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['是', '嘅'], negative=['係'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['乜','他'],negative=['什麼']) # Answer is
    get_anwer_to_analogy_question(zipped_model_list, positive=['企', '走'], negative=['站'])  # Answer is 行
    get_anwer_to_analogy_question(zipped_model_list, positive=['食', '喝'], negative=['吃'])  # Answer is 飲
    get_anwer_to_analogy_question(zipped_model_list, positive=['嗌', '哭'], negative=['叫'])  # Answer is 喊
    get_anwer_to_analogy_question(zipped_model_list, positive=['叫', '喊'], negative=['嗌'])  # Answer is 哭
    get_anwer_to_analogy_question(zipped_model_list, positive=['總統', '中國'], negative=['美國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['主席', '美國'], negative=['中國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['民主', '專政'], negative=['自由'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['自由', '獨裁'], negative=['民主'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['壽司','美國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['點心', '台灣'], negative=['香港'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['壽司', '中國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文','俄羅斯'],negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文', '德國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['粵語', '日本'], negative=['香港'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['東京', '法國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['紐約', '中國'], negative=['美國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['美金', '香港'], negative=['美國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['月亮', '男'], negative=['太陽'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['靚仔', '女人'], negative=['男人'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['男朋友', '靚女'], negative=['靚仔'])


def test_doesnt_match(zipped_model_list):
    print('doesnt match:')
    get_word_doesnt_match(zipped_model_list, "紅 綠 藍 火".split())
    get_word_doesnt_match(zipped_model_list, "星星 月亮 太陽 石頭".split())
    get_word_doesnt_match(zipped_model_list, "魚 蝙蝠 蛇 梨".split())
    get_word_doesnt_match(zipped_model_list, "俄羅斯 美國 印度 導彈".split())
    get_word_doesnt_match(zipped_model_list, "食 飲 吞 跳".split())
    get_word_doesnt_match(zipped_model_list, "汽車 飛機 太陽 單車".split())
    get_word_doesnt_match(zipped_model_list,"男人 女人 小孩 豬".split())
    get_word_doesnt_match(zipped_model_list,"貓 狗 馬 空氣".split())
    get_word_doesnt_match(zipped_model_list,"電腦 電話 手機 蘋果".split())


def test_similar_by_word(zipped_model_list):
    similar_words(zipped_model_list,"太陽", 3)
    similar_words(zipped_model_list,"貓", 3)
    similar_words(zipped_model_list,"屄", 3)
    similar_words(zipped_model_list,"仆街", 3)
    similar_words(zipped_model_list, "企", 3)
    similar_words(zipped_model_list,"紅", 3)
    similar_words(zipped_model_list,"蘋果", 3)
    similar_words(zipped_model_list,"小王子",3)
    similar_words(zipped_model_list,"靚仔",3)
    similar_words(zipped_model_list, "電腦", 3)


def test_similarity(zipped_model_list):
    similarity(zipped_model_list, "快速", "迅速")
    similarity(zipped_model_list, "簡單", "容易")
    similarity(zipped_model_list, "朋友", "夥伴")
    similarity(zipped_model_list, "飛機", "機場")
    similarity(zipped_model_list, "關鍵", "重要")
    similarity(zipped_model_list, "太陽", "地球")
    similarity(zipped_model_list, "葡萄酒", "可樂")


def test_predict_center_word(zipped_model_list):
    predict_center_word(zipped_model_list, ["我", "蘋果"])
    predict_center_word(zipped_model_list,["我","電話"])
    predict_center_word(zipped_model_list, ["飛機", "引擎"])
    predict_center_word(zipped_model_list,["男人","女人"])



def test_is_word_in_model(zipped_model_list,words, is_fasttext=False):

    for model_name, model in zipped_model_list:
        print(model_name)
        for word in words:
            print("Look up word: " + word)
            print(word in model)
            if is_fasttext == True:
                print(word in model.wv.vocab)
            print('-----------------------------------------------------')
        break
    print("***************************************************")
