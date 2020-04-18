from gensim.corpora import WikiCorpus
import codecs
import jieba
def generate_wiki():
    jieba.load_userdict("./data/cantondict2.txt")
    path_to_yue_wiki = './text/zh_yuewiki-latest-pages-articles.xml.bz2'
    i = 0
    wiki = WikiCorpus(path_to_yue_wiki,lemmatize=False)
    file = codecs.open('./text/yue_wiki.txt', 'w', 'utf-8')
    token_list = []
    for text in wiki.get_texts():
        for sentence in text:
            tokens = list(jieba.cut(sentence,HMM=True))
            for token in tokens:
                token_list.append(token)
        str_lines = " ".join(token_list) + "\n"
        file.write(str_lines)
        token_list = []
        i += 1
        if(i % 100 == 0):
            print("Save "+str(i) + " articles")
    file.close()
    print("Finished saved " + str(i) + " articles")

if __name__ == '__main__':
    generate_wiki()