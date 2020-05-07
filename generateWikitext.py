#### This script generates txt version of wiki corpus based on the raw dump file of WikiPedia ####
from gensim.corpora import WikiCorpus
import codecs


# Define the path to find the raw Wiki dump file
path_to_yue_wiki = './text/zh_yuewiki-latest-pages-articles.xml.bz2'

def generate_wiki():

    i = 0
    # Use the WikiCorpus API to read text contents from the raw dump file
    wiki = WikiCorpus(path_to_yue_wiki,lemmatize=False)
    file = codecs.open('./text/yue_wiki2.txt', 'w', 'utf-8')
    # Write texts into the new file article by article
    for text in wiki.get_texts():
        str_lines = " ".join(text) + "\n"
        file.write(str_lines)
        i += 1
        if(i % 100 == 0):
            print("Save "+str(i) + " articles")
    file.close()
    print("Finished saved " + str(i) + " articles")

if __name__ == '__main__':
    generate_wiki()
