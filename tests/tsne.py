from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def draw(model, n_components=2):
    X = model.wv[model.wv.vocab]
    tsne = TSNE(n_components=n_components, verbose=2)
    print("Training tsne...")
    X_tsne = tsne.fit_transform(X)
    print("Training completed.")
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()
#    print('-----------------------------------------------------')
