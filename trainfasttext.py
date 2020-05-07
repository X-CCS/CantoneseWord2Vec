#### This script trains fastText Models####

from gensim.models import FastText, word2vec
from gensim.models import keyedvectors
import time
import os

# This function is used to configure hyper-parameters for training
def train_fasttext(sentences, embedding_size=100, window=5, sg=1, hs=0, min_ct=2,min_n = 1, max_n = 4,
                   ns_exponent=0.75, negative=15, epoch=50, sample_t=1e-5):
    start_time = time.time()
    ftmodel = FastText(size=embedding_size, window=window, sg=sg, hs=hs, negative=negative, sample=sample_t,
                       ns_exponent=ns_exponent,min_n=min_n, max_n = max_n, min_count=min_ct, workers=12, seed=7)
    ftmodel.build_vocab(sentences=sentences)
    ftmodel.train(sentences=sentences,epochs=epoch, total_examples=ftmodel.corpus_count)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    return ftmodel

# This function saves trained fastText model
def save_fasttext(ftmodel,model_path):
    ftmodel.save(model_path)

# This function saves the word vector entities of trained model
def save_fasttext_wv(ftmodel,vector_path):
    ftmodel.wv.save(vector_path)

# This function loads trained fastText model
def load_fasttext(model_path):
    model = FastText.load(model_path)
    return model

# This function loads the word vector entities of trained model
def load_fasttextVectors(vector_path):
    wv = keyedvectors.FastTextKeyedVectors.load(vector_path)
    return wv


if __name__=='__main__':

    #file_path = "./segmentednew"
    #sentences = word2vec.PathLineSentences(file_path)

    # Defined the path to load training data
    file_path = "merged/mergedtext_full_wiki_fictions_new.txt"
    sentences = word2vec.LineSentence(file_path)

    # Set embedding dimension
    emb_dim = 100
    #emb_dim = 50
    #emb_dim = 150
    #emb_dim = 200

    # Set number of negative samples
    # num_ns = 15
    # num_ns = 10
    num_ns = 20

    # Set threshold value of minCount
    min_count = 5

    # Set number of epochs
    #num_epoch = 5
    #num_epoch = 20
    #num_epoch = 50
    num_epoch = 80
    #num_epoch = 110

    # Set subsampling rate on frequent words
    subsample = 0.75
    #subsample = -0.5

    # Set subsampling rate
    sample_rate = 1e-5
    #sample_rate = 1e-4
    #sample_rate = 1e-3

    # Set window size
    # window_size = 3
    # window_size = 5
    window_size = 7
    # window_size = 9

    # Define the output path for trained models
    path_prefix = "F:/fasttextoutput/80epochs/ns_20/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)


    # Train hierarchical softmax model
    print("Training model_hs_{}...".format(window_size))
    model_hs = train_fasttext(sentences, embedding_size=emb_dim, window=window_size, sg=1, hs=1, min_ct=min_count,
                                epoch=num_epoch, sample_t=sample_rate)
    model_name = "model_hs_" + str(window_size) + ".model"
    save_fasttext(model_hs, path_prefix+model_name)
    print("number of words in vocab: %d" % (len(model_hs.wv.vocab)))
    print("{} saved.".format(model_name))

    # Train negative sampling model
    print("Training model_ns_{}...".format(window_size))
    model_ns = train_fasttext(sentences, embedding_size=emb_dim, window=window_size, sg=1, hs=0, min_ct=min_count,
                                negative=num_ns, epoch=num_epoch, ns_exponent=subsample, sample_t=sample_rate)
    model_name = "model_ns_" + str(window_size) + ".model"
    save_fasttext(model_ns, path_prefix+model_name)
    print("{} saved.".format(model_name))