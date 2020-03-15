from gensim.models import word2vec

file_path = "text/training.txt"

sentences = word2vec.LineSentence(file_path)

// Set embedding dimension
emb_dim = 100

// Set number of negative samples
num_ns = 5

// hierarchical softmax with window range of 5
model_hs_5 = word2vec.Word2Vec(sentences,hs=1,min_count=1,window=5,size=emb_dim)
model_hs_5.save("models/model_hs_5.model")

// hierarchical softmax with window range of 10
model_hs_10 = word2vec.Word2Vec(sentences,hs=1,min_count=1,window=10,size=emb_dim)
model_hs_10.save("models/model_hs_10.model")

// negative sampling with window range of 5
model_ns_5 = word2vec.Word2Vec(sentences,hs=0,negative=num_ns,min_count=1,window=5,size=emb_dim)
model_ns_5.save("models/model_ns_5.model")

// negative sampling with window range of 10
model_ns_10 = word2vec.Word2Vec(sentences,hs=0,negative=num_ns,min_count=1,window=10,size=emb_dim)
model_ns_10.save("models/model_ns_10.model")
