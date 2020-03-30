import sys
import os
import codecs
pycan_path = 'data/dict4pku.txt'
output_path = 'splitted2/pycansplitted.txt'
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
                corpus.append(''.join(sentence))
                sentence = []
        for line in corpus:
            o.write(line + '\n')
        o.close()

