import sys
import jieba
import os
import codecs
import re
from zhon.hanzi import punctuation
def read_file(file_path):
    with codecs.open(file_path,"r","utf-8") as file:
        corpus = file.readlines()
        newcorpus = []
        file.close()
        for line in corpus:
            newcorpus.append(line.rstrip("\n"))
        return newcorpus

def save_file(file_path,content):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(file_path,'w','utf-8') as file:
        for line in content:
            file.write(line + '\n')
        file.close()

def clean(line):
    decimal_alpha_cut = re.compile(r"[A-Za-z0-9]|/d+")
    zh_punct_cut = re.compile(r"[%s]+"%punctuation)
    en_punct_cut = re.compile(r"[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[．─⋯]+")
    space_cut = re.compile(r"\s+")
    line = decimal_alpha_cut.sub(r"", line)
    line = zh_punct_cut.sub(r"",line)
    line = en_punct_cut.sub(r"",line)
    line = space_cut.sub(r"",line)
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

corpus_path = "splitted3/"
#corpus_path = "WikiCorpus"
output_path = "segmented_oneline/"
#output_path = "merged2/"
stopword_path = "data/stopwordCT.txt"

def segment():
    jieba.load_userdict("./data/cantondict2.txt")
    jieba.enable_paddle()  # 启动paddle模式。
    with codecs.open(stopword_path,'r',"utf-8") as stopwords:
        stopword_list = [line.strip() for line in stopwords]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(corpus_path)
    for file_name in file_list:
        full_file_path = corpus_path+file_name
        file_name = re.sub(r'.txt',r'',file_name)
        output_file_path = output_path+file_name+'_segmented_paddle.txt'
        corpus = read_file(full_file_path)
        new_corpus = []
        for line in corpus:
            line = clean(line)
            #print(line)
            if line == '':
                continue
            line = list(jieba.cut(line,use_paddle=True))
            #line = filter_stopwords(line,stopword_list)
            if len(line) == 0 :
                continue
            line = ' '.join(line)
            line = line.rstrip('\r\n').strip()
            new_corpus.append(line)
            #print(line)
        save_file(output_file_path,new_corpus)

def mergefiles(filename):
    file_list = os.listdir(output_path)
    print(file_list)
    merged_file_path = 'merged/'
    merged_file_name = merged_file_path + filename
    if not os.path.exists(merged_file_path):
        os.mkdir(merged_file_path)
    with codecs.open(merged_file_name, 'w', 'utf-8') as mergedfile:
        for file_name in file_list:
            full_file_path = output_path + file_name
            with codecs.open(full_file_path, 'r', 'utf-8') as currentfile:
                content = currentfile.read()
                currentfile.close()
                mergedfile.write(content)
        mergedfile.close()
#clean(" 这句话里有英语a和字母1需要被去除，还有中文 标点 符号。。。（））‘’“”【】……,还有英语标点符号(){}[]......\'\"!?/\\\「大作一號」english英语数字1234去除了吗？ ")
#filter_stopwords('','data/stopwordCT.txt')
segment()
#mergefiles('mergedtext_full_wiki_fictions.txt')