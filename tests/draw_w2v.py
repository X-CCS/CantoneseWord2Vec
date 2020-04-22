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

    # pca = PCA(n_components=30)
    # time_start = time.time()
    # vectors_pca = pca.fit_transform(all_vectors)
    #print("PCA done training, spend time: {} seconds".format(time.time() - time_start))
    #tSne = TSNE(n_components=dimension, random_state=None, learning_rate=200, perplexity=10)
    tSne = TSNE(n_components=dimension, init='pca', learning_rate=200,n_iter=2000,random_state=25,perplexity=20)
    print("Training tsne...")
    time_start = time.time()
    vectors_2d = tSne.fit_transform(all_vectors)
    print("Tsne training completed. Spend time: {} seconds.".format(time.time() - time_start))

    x_cors = [v[0] for v in vectors_2d]
    y_cors = [v[1] for v in vectors_2d]

    return x_cors,y_cors,all_vocab,word_dict


def draw_random_annotation(model, n_components=2):
    _rebuild()
    random.seed(25)
    X, Y, labels = train_tsne(model,n_components)

    plt.figure(figsize=(16, 16))
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X, Y)
    num_vocab = len(labels)
    indexes = random.sample(range(num_vocab), 200)
    for i in indexes:
        x = X[i]
        y = Y[i]
        print(labels[i], x, y)
        plt.annotate(labels[i], (x, y), fontsize=14)

    plt.show()

def draw_all(model, n_components = 2):
    _rebuild()
    X, Y, _ = train_tsne(model,n_components)
    plt.figure(figsize=(12, 12))
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X, Y)
    plt.show()

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
    # dim1,dim2,dim3 = np.shape(vectors_groups)
    # vectors_list = np.reshape(vectors_groups,(dim1*dim2,dim3))
    # tSne = TSNE(n_components=2, random_state=None, learning_rate=200)
    # np.set_printoptions(suppress=True)
    # vectors_2d = tSne.fit_transform(vectors_list)
    # vectors_2d = np.reshape(vectors_2d,(dim1,dim2,2))
    for word_group,vectors_group,color in zip(word_groups,vectors_groups,colors):
        tSne = TSNE(n_components=2, random_state=None, learning_rate=200)
        np.set_printoptions(suppress=True)
        vectors_2d = tSne.fit_transform(vectors_group)
        x_coor = [v[0] for v in vectors_2d]
        y_coor = [v[1] for v in vectors_2d]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor,y_coor,color=color)
        for label,x,y in zip(word_group, x_coor,y_coor):
            plt.annotate(label, (x,y))
    plt.show()

def draw_similar_words2(model, words, topn):
    if len(words) > 5:
        print("Support at most 5 words")
        return
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
    for word_group,index_group,color in zip(word_groups,index_groups,colors):
        x_coor = [x_coors[index] for index in index_group]
        y_coor = [y_coors[index] for index in index_group]
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x_coor,y_coor,color=color)
        for label, x, y in zip(word_group, x_coor,y_coor):
            plt.annotate(label, (x,y))
    plt.show()


def draw_word_pairs(model, wordpiars):
    _rebuild()
    if len(wordpiars) > 4:
        print('Only support 4 words at most.')
        return
    vectors = []
    word_labels = []
    # Get topN close words
    for word in wordpiars:
        word_labels.append(word)
        vectors.append(model.wv[word])
    vectors = np.asarray(vectors)
    word_labels = np.asarray(word_labels)
    tSne = TSNE(n_components=2, random_state=None, learning_rate=200)
    np.set_printoptions(suppress=True)
    vectors_2d = tSne.fit_transform(vectors)
    x_coor = [v[0] for v in vectors_2d]
    y_coor = [v[1] for v in vectors_2d]
    plt.rcParams['axes.unicode_minus'] = False
    if len(x_coor) != len(y_coor):
        print('Error')
        return
    i = 0
    while i < len(x_coor):
        if i < 2:
            color = 'blue'
        else:
            color = 'red'
        plt.scatter(x_coor[i], y_coor[i], color=color)
        i += 1
    for label, x, y in zip(word_labels, x_coor, y_coor):
        plt.annotate(label, (x,y))
    plt.show()


def draw_word_pairs2(model, wordpiars):
    _rebuild()
    if len(wordpiars) > 4:
        print('Only support 4 words at most.')
        return
    x_coors,y_coors,all_words,word_dict = train_tsne(model,2)
    word_labels = []
    index_group = []
    # Get topN close words
    for word in wordpiars:
        word_labels.append(word)
        index_group.append(word_dict[word])
        print(all_words[word_dict[word]])


    word_labels = np.asarray(word_labels)
    index_group = np.asarray(index_group)

    x_coor = [x_coors[index] for index in index_group]
    y_coor = [y_coors[index] for index in index_group]
    plt.rcParams['axes.unicode_minus'] = False
    if len(x_coor) != len(y_coor):
        print('Error')
        return
    i = 0
    while i < len(x_coor):
        if i < 2:
            color = 'blue'
        else:
            color = 'red'
        plt.scatter(x_coor[i], y_coor[i], color=color)
        i += 1

    for word, x, y in zip(word_labels, x_coor, y_coor):
        plt.annotate(word, (x,y))
    plt.show()