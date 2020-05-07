#### This script split big chunks text in raw materials into sentences ####
#### using regular expression operations                               ####

import os
import re

# This function takes a paragraph of text as input and output splitted sentences
def cut_sent(para):
    # Seperate sentences without quotation mark #
    para = re.sub(r"\s+",r'',para)
    # para = re.sub('([。！？\?])([^」])', r'\1\n\2', para) # Sentence ends with 。！？
    para = re.sub('([。！])([^」])', r'\1\n\2', para) # Sentence ends with 。and ！
    para = re.sub('(\.{6})([^」])', r'\1\n\2', para) # Sentence ends with ......
    para = re.sub('(\…{2})([^」])', r'\1\n\2', para) # Sentence ends with ……
    # Seperate sentences with quotation mark #
    para = re.sub('([。！？\?][」])([^。！？\?])', r'\1\n\2', para) # Sentence ends with 。！？ and 」
    para = re.sub('([\…{2}][」])([^\…{2}])', r'\1\n\2', para) # Sentence ends with ...... and 」
    para = re.sub('([\.{6}][」])([^\.{6}])', r'\1\n\2', para) # Sentence ends with …… and 」
    para = re.sub('([^。！？\?][」])([^」])',r'\1\n\2',para)  # Sentence ends with 」
    para = para.rstrip()  # Remove spaces at the end of sentences
    return para.split("\n")

# This function takes an article and splits it into sentences
def splitfile (inputfile_path,newfile_path):
    directory = os.path.dirname(inputfile_path)

    if not os.path.exists(directory):

        os.makedirs(directory)

    with open(inputfile_path, 'r',encoding='utf8') as input:
        input_lines = input.readlines()
        with open(newfile_path, 'w', encoding='utf8') as output:
            for paragraph in input_lines:
                newlines = cut_sent(paragraph)
                for newline in newlines:
                    newline += '\n'
                    output.write(newline)


splitfile('text/女神.txt', 'splitted3/nvsheng.txt')
