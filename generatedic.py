#### This Script generates a Cantonese Common Words Dictionary for word segmentation ####

import pycantonese as pc
import os


#load HkCancor Corpus
corpus = pc.hkcancor()

#Get the frequency of words in the corpus
freq = corpus.word_frequency()

## This function takes in an intial dictionary and generate a new dictionary
## New dictionary inherits words from the old dictionary
def save_dict(file_path, init_words_path, tagged_words):
    #Make the directory for saving the generated dictionary
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):

        os.makedirs(directory)
    # Open and read the initial dictionary
    with open(init_words_path, 'r') as t:

        lines = t.readlines()
        # Read words from the corpus and write into the new file
        with open(file_path, 'w', encoding='utf8') as f:

            for word in tagged_words:

                word_line = word[0]

                f.write(word_line + '\n') # One word on each line

            for line in lines:

                f.write(line)

save_dict('data/dict4pku.txt', 'data/init_dict.txt', corpus.tagged_words())