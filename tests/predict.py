from gensim.models import Word2Vec

def predict(model, context, topn=5):
    print(model.predict_output_word(context, topn))
    print('-----------------------------------------------------')
