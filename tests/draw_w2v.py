#### This script has the functions to plot 2D plots of word embeddings

from gensim.models import Word2Vec,KeyedVectors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib.font_manager import _rebuild
import matplotlib.cm as cm
from collections import defaultdict

# Fit the tSNE model based on all word vectors in a model
def train_tsne(model, dimension):
    all_vectors = []
    all_vocab = []
    word_dict = defaultdict(int)
    index = 0
    for word in model.wv.vocab:
        all_vectors.append(model.wv[word])
        all_vocab.append(word)
        word_dict[word] = index
        index += 1
    all_vocab = np.asarray(all_vocab)
    all_vectors = np.asarray(all_vectors)
    tSne = TSNE(n_components=dimension, init='pca', learning_rate=200,n_iter=2000,random_state=None,perplexity=20, n_jobs=12)
    print("Training tsne...")
    time_start = time.time()
    vectors_2d = tSne.fit_transform(all_vectors)
    print("Tsne training completed. Spend time: {} seconds.".format(time.time() - time_start))

    x_cors = [v[0] for v in vectors_2d]
    y_cors = [v[1] for v in vectors_2d]

    return x_cors,y_cors,all_vocab,word_dict

# Plot 2D representations of all words in the vocabulary and label a random selection of words with annotations
def draw_random_annotation(model, n_components=2,sample = 100):
    _rebuild()
    X, Y, vocabs, diction = train_tsne(model,n_components)
    plt.figure(figsize=(20, 20))
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X, Y)
    num_vocab = len(vocabs)
    indexes = random.sample(range(num_vocab), sample)
    for i in indexes:
        x = X[i]
        y = Y[i]
        plt.annotate(vocabs[i], (x, y), fontsize=20)

    plt.show()


# Plot 2D representations of all words in the vocabulary
def draw_all(model, n_components = 2):
    _rebuild()
    X, Y, _ = train_tsne(model,n_components)
    plt.figure(figsize=(12, 12))
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X, Y)
    plt.show()



# This function plots the 2D representations of a group of words
# together with their top n similar words
# the tSNE model fits on each group of similar word groups
def draw_similar_words(model, words, topn):
    if len(words) > 5:
        print("Support at most 5 words")
        return
    _rebuild()
    vectors_groups = []
    vectors = []
    word_groups = []
    word_labels = []
    # Get topN close words
    for word in words:
        word_labels.append(word)
        vectors.append(model.wv[word])
        close_words = model.similar_by_word(word,topn=topn)
        for close_word in close_words:
            word_labels.append(close_word[0])
            vectors.append(model.wv[close_word[0]])
        vectors_groups.append(vectors)
        word_groups.append(word_labels)
        vectors = []
        word_labels = []
    vectors_groups = np.asarray(vectors_groups)
    word_groups = np.asarray(word_groups)
    colors = cm.rainbow(np.linspace(0, 1, len(words)))
    for word_group,vectors_group,color in zip(word_groups,vectors_groups,colors):
        tSne = TSNE(n_components=2, init='pca', learning_rate=200,n_iter=2000,random_state=None,perplexity=20)
        np.set_printoptions(suppress=True)
        vectors_2d = tSne.fit_transform(vectors_group)
        x_coor = [v[0] for v in vectors_2d]
        y_coor = [v[1] for v in vectors_2d]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor,y_coor,color=color)
        for label,x,y in zip(word_group, x_coor,y_coor):
            plt.annotate(label, (x,y))
    plt.show()


# This function plots the 2D representations of a group of words
# together with their top n similar words
# the tSNE model fits on the whole vocabulary
def draw_similar_words2(model, words, topn):
    _rebuild()
    x_coors,y_coors,all_words,word_dict = train_tsne(model,2)
    word_groups = []
    word_labels = []
    index_group = []
    index_groups = []
    # Get topN close words
    for word in words:
        word_labels.append(word)
        index_group.append(word_dict[word])
        close_words = model.similar_by_word(word,topn=topn)
        for close_word in close_words:
            word_labels.append(close_word[0])
            index = word_dict[close_word[0]]
            index_group.append(index)
        word_groups.append(word_labels)
        index_groups .append(index_group)
        word_labels = []
        index_group = []
    index_groups = np.asarray(index_groups)
    word_groups = np.asarray(word_groups)
    colors = cm.rainbow(np.linspace(0, 1, len(words)))
    plt.figure(figsize=(14, 14))
    for word_group,index_group,color in zip(word_groups,index_groups,colors):
        x_coor = [x_coors[index] for index in index_group]
        y_coor = [y_coors[index] for index in index_group]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor,y_coor,color=color)
        for label, x, y in zip(word_group, x_coor,y_coor):
            plt.annotate(label, (x,y), fontsize=14)
    plt.show()

# This function plots the 2D representations of a group of words
# together with their top n similar words
# the tSNE model fits the selected words and their similar words
def draw_similar_words3(model, words, topn):
    _rebuild()
    vectors_groups = []
    vectors = []
    word_groups = []
    word_labels = []
    # Get topN close words
    for word in words:
        word_labels.append(word)
        vectors.append(model.wv[word])
        close_words = model.similar_by_word(word,topn=topn)
        for close_word in close_words:
            word_labels.append(close_word[0])
            vectors.append(model.wv[close_word[0]])
        vectors_groups.append(vectors)
        word_groups.append(word_labels)
        vectors = []
        word_labels = []
    vectors_groups = np.asarray(vectors_groups)
    word_groups = np.asarray(word_groups)
    vectors_groups_flatter = np.asarray([v for v_group in vectors_groups for v in v_group])
    tSne = TSNE(n_components=2, init='pca', learning_rate=200,n_iter=2000,random_state=None,perplexity=20)
    np.set_printoptions(suppress=True)
    vectors_2d = tSne.fit_transform(vectors_groups_flatter)
    colors = cm.rainbow(np.linspace(0, 1, len(words)))
    x_coords = vectors_2d[:, 0]
    y_coords = vectors_2d[:, 1]
    plt.figure(figsize=(14, 14))
    for i in range(len(words)):
        x_coor = [x for x in x_coords[i*(topn+1):(i+1)*(topn+1)]]
        y_coor = [y for y in y_coords[i*(topn+1):(i+1)*(topn+1)]]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor, y_coor, color=colors[i])
        for label,x,y in zip(word_groups[i], x_coor,y_coor):
            plt.annotate(label, (x,y), fontsize=14)
    plt.show()



# This function plots the 2D representations of two paris of related words
def draw_word_pairs3(model, wordpairs):
    _rebuild()
    if len(wordpairs) > 5:
        print('Only support 5 pair of words at most.')
        return
    vectors_groups = []
    vectors = []
    word_groups = []
    word_labels = []

    for wordpair in wordpairs:
        for word in wordpair:
            word_labels.append(word)
            vectors.append(model.wv[word])
        vectors_groups.append(vectors)
        word_groups.append(word_labels)
        vectors = []
        word_labels = []
    vectors_groups = np.asarray(vectors_groups)
    word_groups = np.asarray(word_groups)
    vectors_groups_flatten = np.asarray([v for v_group in vectors_groups for v in v_group])
    tSne = TSNE(n_components=2, init='pca', learning_rate=200,n_iter=3000,random_state=None,perplexity=20)
    np.set_printoptions(suppress=True)
    vectors_2d = tSne.fit_transform(vectors_groups_flatten)
    colors = cm.rainbow(np.linspace(0, 1, len(wordpairs)))
    for i in range(len(wordpairs)):
        x_coor = [v[0] for v in vectors_2d[i*(2):(i+1)*(2)]]
        y_coor = [v[1] for v in vectors_2d[i*(2):(i+1)*(2)]]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor, y_coor, color=colors[i])
        for label,x,y in zip(word_groups[i], x_coor,y_coor):
            plt.annotate(label, (x,y))
    plt.show()