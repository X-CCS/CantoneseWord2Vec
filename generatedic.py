import pycantonese as pc

import os

import re

import codecs

corpus = pc.hkcancor()#load HkCancor Corpus

freq = corpus.word_frequency()

def save(file_path, init_words_path, tagged_words):

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):

        os.makedirs(directory)

    with codecs.open(init_words_path, 'r', 'utf8') as t:

        lines = t.readlines()

        with codecs.open(file_path, 'w', 'utf8') as f:

            for word in tagged_words:

                word_freq = freq[word[0]] if word[0] in freq else None

                word_tag = word[1].lower()

                word_tag_matched = bool(re.match('^[a-z]+$', word_tag))

                word_line = word[0]

                if word_freq is not None:

                    word_line = word_line + ' ' + str(word_freq)

                if word_tag_matched is True:

                    word_line = word_line + ' ' + str(word_tag)

                f.write(word_line + '\n')

            for line in lines:

                f.write(line)

save('data/cantondict2.txt', 'data/init_dict.txt', corpus.tagged_words())