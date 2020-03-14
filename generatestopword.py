import pycantonese as pc

import os

import re

corpus = pc.hkcancor()#load HkCancor Corpus

stop = pc.stop_words()
def save(file_path, init_words_path, tagged_words):

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):

        os.makedirs(directory)


    with open(file_path, 'w', encoding='utf8') as f:

        for word in tagged_words:

            word_line = word

            f.write(word_line + '\n')

save('data/stopwordCT.txt', 'data/init_dict.txt', stop)