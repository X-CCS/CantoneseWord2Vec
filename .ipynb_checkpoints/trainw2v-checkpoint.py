from gensim.models import word2vec
from gensim.models import keyedvectors
import time

def train_word2vec(sentences, embedding_size=100, window=5, sg=1, hs=0, min_c=2, ns_exponent=0.75, negative = 20,
                   epoch = 50, sample = 1e-5):
    w2vmodel = word2vec.Word2Vec(sentences, size=embedding_size, window=window, sg=sg, hs=hs,
                                 negative=negative, sample=sample, ns_exponent=ns_exponent,
                                 min_count=min_c, workers=8, iter=epoch, compute_loss=True, seed=7)
    return w2vmodel


def save_word2vec(w2vModel,model_path,vector_path):
    w2vModel.save(model_path)
    w2vModel.wv.save(vector_path)


def load_w2vModel(model_path):
    model = word2vec.Word2Vec.load(model_path)
    return model

def load_w2vVectors(vector_path):
    wv = keyedvectors.Word2VecKeyedVectors.load(vector_path)
    return wv

if __name__=='__main__':

    file_path = "./segmentednew"
    #file_path = "./segmented_oneline"

    sentences = word2vec.PathLineSentences(file_path)

    #file_path = "merged/mergedtext_full_wiki_fictions_new.txt"
    # file_path = "merged/mergedtext_full_oneline_wiki_fiction.txt"

    #sentences = word2vec.LineSentence(file_path)

    # Set embedding dimension
    emb_dim = 100

    # Set number of negative samples
    num_ns = 20

    # Set threshold value of minCount
    min_count = 5

    # Set number of epochs
    num_epoch = 20

    # hierarchical softmax with window range of 5
    print("Training skip gram model_hs_5...")
    start_time = time.time()
    model_hs_5 = train_word2vec(sentences, embedding_size=emb_dim, window=5, sg=1, hs=1, min_c=min_count,
                                epoch=num_epoch)
    training_time = time.time() - start_time
    print("%d seconds used to train this model\n" %(training_time))
    save_word2vec(model_hs_5, "models/model_hs_5.model", "models/vector_hs_5.kv")
    print("number of words in volcab: %d\n" % (len(model_hs_5.wv.vocab)))
    print("models/model_hs_5.model saved.")

    # hierarchical softmax with window range of 10
    print("Training model_hs_10...")
    start_time = time.time()
    model_hs_10 = train_word2vec(sentences, embedding_size=emb_dim, window=10, sg=1, hs=1, min_c=min_count,
                                 epoch=num_epoch)
    training_time = time.time() - start_time
    print("%d seconds used to train this model\n" %(training_time))
    save_word2vec(model_hs_10, "models/model_hs_10.model", "models/vector_hs_10.kv")
    print("models/model_hs_10.model saved.")

    # negative sampling with window range of 5
    print("Training model_ns_5...")
    start_time = time.time()
    model_ns_5 = train_word2vec(sentences, embedding_size=emb_dim, window=5, sg=1, hs=0, min_c=min_count,
                                negative=num_ns, epoch=num_epoch)
    training_time = time.time() - start_time
    print("%d seconds used to train this model\n" %(training_time))
    save_word2vec(model_ns_5, "models/model_ns_5.model", "models/vector_ns_5.kv")
    print("models/model_ns_5.model saved.")

    # negative sampling with window range of 10
    print("Training model_ns_10...")
    start_time = time.time()
    model_ns_10 = train_word2vec(sentences, embedding_size=emb_dim, window=10, sg=1, hs=0, min_c=min_count,
                                negative=num_ns, epoch=num_epoch)
    training_time = time.time() - start_time
    print("%d seconds used to train this model\n" %(training_time))
    save_word2vec(model_ns_10, "models/model_ns_10.model", "models/vector_ns_10.kv")
    print("models/model_ns_10.model saved.")