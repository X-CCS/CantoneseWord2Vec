#### This script performs text pre-processing on splitted corpus ####
import jieba
import os
import codecs
import re
from zhon.hanzi import punctuation

# This function load the raw corpus file which is splitted into sentences
def read_file(file_path):
    with codecs.open(file_path,"r","utf-8") as file:
        corpus = file.readlines()
        newcorpus = []
        file.close()
        for line in corpus:
            newcorpus.append(line.rstrip("\n"))
        return newcorpus

# This function save the processed corpus into text file
def save_file(file_path,content):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(file_path,'w','utf-8') as file:
        for line in content:
            file.write(line + '\n')
        file.close()

# This function clean the input sentence to remove punctuations, digits and foreign characters
def clean(line):
    zh_chars = re.findall('[\n\s*\r\u4e00-\u9fa5]', line)
    line = "".join(zh_chars)
    zh_punct_cut = re.compile(r"[%s]+"%punctuation)
    en_punct_cut = re.compile(r"[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[．─⋯]+")
    space_cut = re.compile(r"\s+")
    line = zh_punct_cut.sub(r"",line)
    line = en_punct_cut.sub(r"",line)
    line = space_cut.sub(r"",line)
    return line

# This function filter out stop words from input sentences
def filter_stopwords(line,stopword_list):
    line_filtered = []
    for word in line:
        if word in stopword_list:
            continue
        else:
            line_filtered.append(word)
    return line_filtered

corpus_path = "splitted/"
#corpus_path = "WikiCorpus/"
output_path = "segmentednew/"
#output_path = "merged2/"
stopword_path = "data/stopwordCT.txt"


# This function calls other functions to clean and segment splitted corpus
def segment():
    jieba.load_userdict("./data/cantondict2.txt")
    # Read in a list stopwords
    with codecs.open(stopword_path,'r',"utf-8") as stopwords:
        stopword_list = [line.strip() for line in stopwords]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(corpus_path)
    for file_name in file_list:
        full_file_path = corpus_path+file_name
        file_name = re.sub(r'.txt',r'',file_name)
        output_file_path = output_path+file_name+'_segmented_full.txt'

        # Load text files with separate sentences
        corpus = read_file(full_file_path)
        new_corpus = []

        # Clean and segment sentences line by line
        for line in corpus:
            # Clean the sentence
            line = clean(line)
            if line == '':
                continue

            # Segment the sentence
            line = list(jieba.cut(line,HMM=True))
            #line = filter_stopwords(line,stopword_list)
            if len(line) == 0 :
                continue
            # Join the segmented words with spaces
            line = ' '.join(line)
            line = line.rstrip('\r\n').strip()
            new_corpus.append(line)
        save_file(output_file_path,new_corpus)

# This function merge seperate segmented articles into one text file
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

if __name__ == '__main__':
    segment()  # Segment the reformated corpus
    mergefiles('mergedtext_full_wiki_fictions_new.txt') # Merge segmented articles