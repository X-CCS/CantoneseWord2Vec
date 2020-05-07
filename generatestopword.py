#### The scripts generates a stop word dictionary for Cantonese Corpus ####
import pycantonese as pc

import os

#load HkCancor Corpus
corpus = pc.hkcancor()

stop = pc.stop_words()

## This function generate the stop word dictionary using the stops provided in HkCancor corpus
def save_stopwords(file_path, tagged_words):

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):

        os.makedirs(directory)


    with open(file_path, 'w', encoding='utf8') as f:

        for word in tagged_words:

            word_line = word

            f.write(word_line + '\n')

save_stopwords('data/stopwordCT.txt', 'data/init_dict.txt', stop)