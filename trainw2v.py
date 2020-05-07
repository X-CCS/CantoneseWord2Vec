#### This script trains Word2Vec Models####

from gensim.models import word2vec
from gensim.models import keyedvectors
import time
import os

# This function is used to configure hyper-parameters for training
def train_word2vec(sentences, embedding_size=100, window=5, sg=1, hs=0, min_c=2, ns_exponent=0.75, negative = 20,
                   epoch = 50, sample_t = 1e-5):
    start_time = time.time()
    w2vmodel = word2vec.Word2Vec(sentences, size=embedding_size, window=window, sg=sg, hs=hs,
                                 negative=negative, sample=sample_t, ns_exponent=ns_exponent,
                                 min_count=min_c, workers=12, iter=epoch, compute_loss=True, seed=7)
    training_time = time.time() - start_time
    print("%d seconds used to train this model" %(training_time))
    return w2vmodel

# This function saves trained Word2Vec model
def save_word2vec(w2vModel,model_path):
    w2vModel.save(model_path)

# This function saves the word vector entities of trained model
def save_word2vec_wv(w2vModel,vector_path):
    w2vModel.wv.save(vector_path)

# This function loads trained Word2Vec model
def load_w2vModel(model_path):
    model = word2vec.Word2Vec.load(model_path)
    return model

# This function loads the word vector entities of trained model
def load_w2vVectors(vector_path):
    wv = keyedvectors.Word2VecKeyedVectors.load(vector_path)
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

    # Set negative sampling distribution parameter
    negsample = 0.75
    #negsample = -0.5

    # Set subsampling rate
    sample_rate = 1e-5
    #sample_rate = 1e-4
    #sample_rate = 1e-3

    # Set window size
    window_size = 3
    # window_size = 5
    # window_size = 7
    # window_size = 9

    # Define the output path for trained models
    path_prefix = "F:/word2vecOutput/80epochs/ns_20/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    # Train hierarchical softmax model
    print("Training model_hs_{}...".format(window_size))
    model_hs = train_word2vec(sentences, embedding_size=emb_dim, window=window_size, sg=1, hs=1, min_c=min_count,
                                epoch=num_epoch, sample_t=sample_rate)
    model_name = "model_hs_" + str(window_size) + ".model"
    save_word2vec(model_hs, path_prefix+model_name)
    print("number of words in volcab: %d" % (len(model_hs.wv.vocab)))
    print("{} saved.".format(model_name))

    # Train negative sampling model
    print("Training model_ns_{}...".format(window_size))
    model_ns_3 = train_word2vec(sentences, embedding_size=emb_dim, window=window_size, sg=1, hs=0, min_c=min_count,
                                negative=num_ns, epoch=num_epoch, ns_exponent=negsample, sample_t=sample_rate)
    model_name = "model_ns_" + str(window_size) + ".model"
    save_word2vec(model_ns_3, path_prefix+model_name)
    print("{} saved.".format(model_name))