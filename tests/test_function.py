#### This script contains function for testing

# This function finds top n similar words of a word
def find_similar_words(model, target, topn=5):
    for word in model.wv.similar_by_word(target,topn=topn):
        print(word[0], word[1])
    print('-----------------------------------------------------')

# This function calculate the similarity between two words
def find_similarity(model, word1, word2):
    print(model.wv.similarity(word1, word2))
    #print(model.wv.distance(word1,word2))
    print('-----------------------------------------------------')

# This function finds the top n answers that can answer a semantic analogy question
def find_answer_analogy_question(model,positive,negative,topn=5):
    print(model.wv.most_similar(positive=positive, negative=negative,topn=topn))
    print('-----------------------------------------------------')

# This function finds the word that doesn't belong to a group of words
def find_doesnt_match(model,words):
    print(model.wv.doesnt_match(words))
    print('-----------------------------------------------------')

# This function get the accuracy of a model based the semantic analogy test sets
def get_accuracy(model,path_to_file):
    accuracy,section_result = model.wv.evaluate_word_analogies(path_to_file)
    num_of_section = len(section_result)
    print("Current model's overall accuracy is {:.2f}%".format(accuracy*100))
    for i in range(num_of_section-1):
        section = section_result[i]
        section_name = section['section']
        section_correct = len(section['correct'])
        section_total = len(section['correct']) + len(section['incorrect'])
        section_accuray = 100*float(section_correct)/section_total
        print('Section {} has an accuracy of {:.2f}%'.format(section_name,section_accuray))
    print('-----------------------------------------------------')

# This function finds the top n similar words of a given word vector
def find_similar_words_by_vector(model, target, topn=5):
    for word in model.wv.similar_by_vector(target,topn=topn):
        print(word[0], word[1])
    print('-----------------------------------------------------')