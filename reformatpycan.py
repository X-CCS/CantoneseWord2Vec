import sys
import os
import codecs
import jieba
import re
from zhon.hanzi import punctuation

pycan_path = 'data/dict4pku.txt'
output_path = 'splitted2/pycansplitted2.txt'
output_path2 = 'segmented/pycansegmented.txt'
stopword_path = "data/stopwordCT.txt"


def reformat():
    with codecs.open(pycan_path,'r','utf-8') as pc:
        pc_lines = pc.readlines()
        pc.close()
        with codecs.open(output_path,'w','utf-8') as o:
            corpus = []
            sentence = []
            for word in pc_lines:
                word = word.rstrip('\r\n')
                if not word in ['.','?','!' ]:
                    if word == '"':
                        continue
                    else:
                        sentence.append(word)
                else:
                    corpus.append(' '.join(sentence))
                    sentence = []
            for line in corpus:
                o.write(line + '\n')
            o.close()


def read_file(file_path):
    with codecs.open(file_path,"r","utf-8") as file:
        corpus = file.readlines()
        newcorpus = []
        file.close()
        for line in corpus:
            newcorpus.append(line.rstrip("\n"))
        return newcorpus


def save_file(file_path,article):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(file_path,'w','utf-8') as file:
        for line in article:
            file.write(line + '\n')
        file.close()


def clean(line):
    decimal_alpha_cut = re.compile(r"[A-Za-z0-9]|/d+")
    zh_punct_cut = re.compile(r"[%s]+"%punctuation)
    en_punct_cut = re.compile(r"[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[．─⋯]+")
    line = decimal_alpha_cut.sub(r"", line)
    line = zh_punct_cut.sub(r"",line)
    line = en_punct_cut.sub(r"",line)
    #print(line)
    return line


def filter_stopwords(line,stopword_list):
    line_filtered = []
    for word in line:
        if word in stopword_list:
            continue
        else:
            line_filtered.append(word)
    return line_filtered


def segment():

    with codecs.open(stopword_path,'r',"utf-8") as stopwords:
        stopword_list = [line.strip() for line in stopwords]
    corpus = read_file(output_path)
    new_corpus = []
    for line in corpus:
        line = clean(line)
        if line == '':
            continue
        if len(line) == 0:
            continue
        line = line.rstrip('\r\n').strip()
        new_corpus.append(line)
    save_file(output_path2,new_corpus)
if __name__ == '__main__':
    #reformat()
    segment()