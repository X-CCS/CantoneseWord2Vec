from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def draw(model, n_components=2):
    W = random.sample(list(model.wv.vocab), 2000)
    X = model.wv[W]
    tsne = TSNE(n_components=n_components, verbose=0)
    print("Training tsne...")
    X_tsne = tsne.fit_transform(X)
    print("Training completed.")
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    
    I = random.sample(range(2000), 200)
    for i in I:
        print(W[i], X_tsne[i][0], X_tsne[i][1])
        plt.annotate(W[i], (X_tsne[i][0], X_tsne[i][1]))
    
#    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()
#    print('-----------------------------------------------------')
