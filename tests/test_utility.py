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
    print("The word that doesn't belongs to the group is?")
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

def test_word_analogies(zipped_model_list):
    get_anwer_to_analogy_question(zipped_model_list, positive=['皇帝', '女'], negative=['男'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['女皇', '男'], negative=['女'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['國王', '女'], negative=['男'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['皇后', '男'], negative=['女'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['皇后', '皇帝'], negative=['女皇'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['台語', '香港'], negative=['台灣'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['是', '嘅'], negative=['係'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['乜','他'],negative=['什麼'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['企', '走'], negative=['站'])  # Answer is 行
    get_anwer_to_analogy_question(zipped_model_list, positive=['食', '喝'], negative=['吃'])  # Answer is 飲
    get_anwer_to_analogy_question(zipped_model_list, positive=['嗌', '哭'], negative=['叫'])  # Answer is 喊
    get_anwer_to_analogy_question(zipped_model_list, positive=['總統', '中國'], negative=['美國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['主席', '美國'], negative=['中國'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['民主', '集權'], negative=['自由'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['民主', '專政'], negative=['自由'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['壽司','美國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['點心', '台灣'], negative=['香港'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['壽司', '中國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['壽司', '香港'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文','俄羅斯'],negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文', '德國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文', '英國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['日文', '法國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['東京', '德國'], negative=['日本'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['美國', '香港'], negative=['美金'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['月亮', '男'], negative=['太陽'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['月亮', '太陽'], negative=['地球'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['靚仔', '她'], negative=['他'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['靚仔', '女人'], negative=['男人'])
    get_anwer_to_analogy_question(zipped_model_list, positive=['男朋友', '靚女'], negative=['靚仔'])

def test_doesnt_match(zipped_model_list):
    print('doesnt match:')
    get_word_doesnt_match(zipped_model_list, "紅 綠 藍 火".split())
    get_word_doesnt_match(zipped_model_list, "星星 月亮 太陽 石頭".split())
    get_word_doesnt_match(zipped_model_list, "魚 蝙蝠 蛇 梨".split())
    get_word_doesnt_match(zipped_model_list, "俄羅斯 美國 印度 導彈".split())
    get_word_doesnt_match(zipped_model_list, "食 飲 吞 跳".split())


def test_similar_by_word(zipped_model_list):
    similar_words(zipped_model_list,"太陽", 3)
    similar_words(zipped_model_list,"貓", 3)
    similar_words(zipped_model_list,"屌", 3)
    similar_words(zipped_model_list,"紅", 3)
    similar_words(zipped_model_list,"蘋果", 3)
    similar_words(zipped_model_list,"中國", 3)
    similar_words(zipped_model_list,"他",3)
    similar_words(zipped_model_list,"小王子",3)
    similar_words(zipped_model_list,"靚仔",3)


def test_similarity(zipped_model_list):
    similarity(zipped_model_list, "紅", "綠")
    similarity(zipped_model_list, "你", "我")


def test_predict_center_word(zipped_model_list):
    predict_center_word(zipped_model_list, ["我", "蘋果"])
    predict_center_word(zipped_model_list, ["飛機", "引擎"])


def test_is_word_in_model(zipped_model_list,word, is_fasttext=False):
    print("Look up word: "+word+"in this model,")
    for model_name, model in zipped_model_list:
        print(model_name)
        print(word in model)
        if is_fasttext == True:
            print(word in model.wv.vocab)
    print("***************************************************")