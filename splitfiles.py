#coding:utf-8
import sys
import os
import re

def cut_sent(para):
    # Seperate sentences without quotation mark #
    para = re.sub('([。！？\?])([^」])', r'\1\n\2', para)
    para = re.sub('(\.{6})([^」])', r'\1\n\2', para)
    para = re.sub('(\…{2})([^」])', r'\1\n\2', para)
    # Seperate sentences with quotation mark #
    para = re.sub('([。！？\?][」])([^。！？\?])', r'\1\n\2', para)
    para = re.sub('([\…{2}][」])([^\…{2}])', r'\1\n\2', para)
    para = re.sub('([\.{6}][」])([^\.{6}])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")

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



#print(cut_sent("當整件事太撲朔迷離嘅時候，原來人係唔會夠膽反抗。喺呢個遠離人煙嘅沙漠度，雖然我愈嚟愈接近死亡，但我竟然都聽佢話，拎咗張紙同埋支鋼筆出嚟。不過我又突然間醒起，我喺學校淨係學過地理、歷史、數學同埋文法，所以就☐☐哋噉同個細路仔講其實我唔識畫畫。佢鼓勵我話： 「唔緊要，幫我畫隻綿羊就得㗎喇。」由於我從來都冇畫過羊，所以將我唯一識畫嘅兩幅畫其中一幅畫咗畀佢。"))
splitfile('text/hzh.txt','text/hzhsplitted.txt')
