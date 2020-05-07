from gensim.models import Word2Vec
# This function predict the central words using the context
def predict(model, context, topn=5):
    print(model.predict_output_word(context, topn))
    print('-----------------------------------------------------')
