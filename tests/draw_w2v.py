from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.font_manager import _rebuild


def draw(model, n_components=2):
    _rebuild()
    random.seed(25)
    words = random.sample(list(model.wv.vocab), 5000)
    vectors = model.wv[words]
    vectors = np.asarray(vectors)
    labels = np.asarray(words)
    tsne = TSNE(n_components=n_components, verbose=0,random_state=0,learning_rate=100)
    print("Training tsne...")
    vectors_tsne = tsne.fit_transform(vectors)
    print("Training completed.")

    plt.figure(figsize=(12, 12))
    plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])

    indexs = random.sample(range(5000), 200)
    for i in indexs:
        x = vectors_tsne[i][0]
        y = vectors_tsne[i][1]
        print(labels[i], x, y)
        plt.text(labels[i], (x, y))

    #    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()
#    print('-----------------------------------------------------')
def draw_all(model, n_components = 2):
    _rebuild()
    vocab = list(model.wv.vocab)
    vectors = []
    words = []
    for word in vocab:
        vectors.append(model.wv[word])
        words.append(word)
    words = np.asarray(words)
    vectors = np.asarray(vectors)

    tsne = TSNE(n_components=n_components, verbose=0, random_state=0, learning_rate=100)
    print("Training tsne...")
    vectors_tsne = tsne.fit_transform(vectors)
    print("Training completed.")

    X = [v[0] for v in vectors_tsne]
    Y = [v[1] for v in vectors_tsne]

    plt.figure(figsize=(12, 12))
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X, Y)
    plt.show()



