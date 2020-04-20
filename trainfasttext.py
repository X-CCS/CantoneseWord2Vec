from gensim.models import FastText, word2vec
from gensim.models import keyedvectors
import time
import os

def train_fasttext(sentences, embedding_size=100, window=5, sg=1, hs=0, min_ct=2,min_n = 2, max_n = 4,
                   ns_exponent=0.75, negative=15, epoch=50, sample=1e-5):
    ftmodel = FastText(size=embedding_size, window=window, sg=sg, hs=hs, negative=negative, sample=sample,
                       ns_exponent=ns_exponent,min_n=min_n, max_n = max_n, min_count=min_ct, workers=8, seed=7)
    ftmodel.build_vocab(sentences=sentences)
    ftmodel.train(sentences=sentences,epochs=epoch, total_examples=ftmodel.corpus_count)
    return ftmodel


def save_fasttext(ftmodel,model_path,vector_path):
    ftmodel.save(model_path)
    ftmodel.wv.save(vector_path)


def load_fasttext(model_path):
    model = FastText.load(model_path)
    return model


def load_fasttextVectors(vector_path):
    wv = keyedvectors.FastTextKeyedVectors.load(vector_path)
    return wv


if __name__=='__main__':

    file_path = "./segmentednew"
    #file_path = "./segmented_oneline"

    sentences = word2vec.PathLineSentences(file_path)

    #file_path = "merged/mergedtext_full_wiki_fictions_new.txt"
    # file_path = "merged/mergedtext_full_oneline_wiki_fiction.txt"
    #sentences = Word2vec.LineSentence(file_path)

    # Set embedding dimension
    emb_dim = 100

    # Set number of negative samples
    num_ns = 15

    # Set threshold value of minCount
    min_count = 5

    # Set number of epochs
    num_epoch = 20
    #num_epoch = 50
    #num_epoch = 80

    # Set subsampling rate on frequent words
    #subsample = 0.75
    subsample = -0.5

    path_prefix = "./models/fasttext/20epochs/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    # hierarchical softmax with window range of 3
    print("Training model_hs_3...")
    start_time = time.time()
    model_hs_3 = train_fasttext(sentences, embedding_size=emb_dim, window=3, sg=1, hs=1, min_ct=min_count,
                                epoch=num_epoch, ns_exponent=subsample)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    save_fasttext(model_hs_3, path_prefix+"model_hs_3.model", path_prefix+"vector_hs_3.kv")
    #print("model_hs_3 has last training loss:%f" % model_hs_3.get_latest_training_loss())
    print("model_hs_3.model saved.")


    # hierarchical softmax with window range of 7
    print("Training skip gram model_hs_7...")
    start_time = time.time()
    model_hs_7 = train_fasttext(sentences, embedding_size=emb_dim, window=7, sg=1, hs=1, min_ct=min_count,
                                epoch=num_epoch, ns_exponent=subsample)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    save_fasttext(model_hs_7, path_prefix+"model_hs_7.model", path_prefix+"vector_hs_7.kv")
    print("number of words in volcab: %d" % (len(model_hs_7.wv.vocab)))
    #print("model_hs_7 has last training loss:%f" % model_hs_7.get_latest_training_loss())
    print("model_hs_7.model saved.")

    # negative sampling with window range of 3
    print("Training model_ns_3...")
    start_time = time.time()
    model_ns_3 = train_fasttext(sentences, embedding_size=emb_dim, window=3, sg=1, hs=0, min_ct=min_count,
                                negative=num_ns, epoch=num_epoch, ns_exponent=subsample)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    save_fasttext(model_ns_3, path_prefix+"model_ns_3.model", path_prefix+"vector_ns_3.kv")
    #print("model_ns_3 has last training loss:%f" % model_ns_3.get_latest_training_loss())
    print("model_ns_3.model saved.")

    # negative sampling with window range of 7
    print("Training model_ns_7...")
    start_time = time.time()
    model_ns_7 = train_fasttext(sentences, embedding_size=emb_dim, window=7, sg=1, hs=0, min_ct=min_count,
                                negative=num_ns, epoch=num_epoch, ns_exponent=subsample)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    save_fasttext(model_ns_7, path_prefix+"model_ns_7.model", path_prefix+"vector_ns_7.kv")
    #print("model_ns_7 has last training loss:%f" % model_ns_7.get_latest_training_loss())
    print("model_ns_7.model saved.")

