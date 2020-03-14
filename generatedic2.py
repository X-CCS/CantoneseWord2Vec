import pycantonese as pc

import os

import re

corpus = pc.hkcancor()#load HkCancor Corpus

freq = corpus.word_frequency()

def save(file_path, init_words_path, tagged_words):

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):

        os.makedirs(directory)

    with open(init_words_path, 'r') as t:

        lines = t.readlines()

        with open(file_path, 'w', encoding='utf8') as f:

            for word in tagged_words:

                word_line = word[0]

                f.write(word_line + '\n')

            for line in lines:

                f.write(line)

save('data/dict4pku.txt', 'data/init_dict.txt', corpus.tagged_words())