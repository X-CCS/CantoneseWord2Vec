from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def draw(model, n_components=2):
    X = model.wv[model.wv.vocab]
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()
#    print('-----------------------------------------------------')
