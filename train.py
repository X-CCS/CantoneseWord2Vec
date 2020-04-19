from gensim.models import word2vec
from gensim.models import keyedvectors

#file_path = "segmented"
#
#sentences = word2vec.PathLineSentences(file_path)

file_path = "merged/mergedtext_full_wiki_fictions_new.txt"
#file_path = "merged/mergedtext_full_oneline_wiki_fiction.txt"

sentences = word2vec.LineSentence(file_path)

# Set embedding dimension
emb_dim = 100

# Set number of negative samples
num_ns = 10

# Set number of epochs
num_epoch = 15

# hierarchical softmax with window range of 5
print("Training model_hs_5...")
model_hs_5 = word2vec.Word2Vec(sentences,hs=1,sg=1,negative=0,min_count=5,window=5,size=emb_dim,iter=num_epoch)
model_hs_5.save("models/model_hs_5.model")
print("models/model_hs_5.model saved.")

## hierarchical softmax with window range of 10
#print("Training model_hs_10...")
#model_hs_10 = word2vec.Word2Vec(sentences,hs=1,sg=1,negative=0,min_count=5,window=10,size=emb_dim,iter=num_epoch)
#model_hs_10.save("models/model_hs_10.model")
#print("models/model_hs_10.model saved.")

# negative sampling with window range of 5
print("Training model_ns_5...")
model_ns_5 = word2vec.Word2Vec(sentences,hs=0,sg=1,negative=num_ns,min_count=5,window=5,size=emb_dim,iter=num_epoch)
model_ns_5.save("models/model_ns_5.model")
print("models/model_ns_5.model saved.")

## negative sampling with window range of 10
#print("Training model_ns_10...")
#model_ns_10 = word2vec.Word2Vec(sentences,hs=0,sg=1,negative=num_ns,min_count=5,window=10,size=emb_dim,iter=num_epoch)
#model_ns_10.save("models/model_ns_10.model")
#print("models/model_ns_10.model saved.")
