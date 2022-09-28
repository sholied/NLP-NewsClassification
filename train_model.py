import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import modelLSTM
import matplotlib.pyplot as plt
from DataPreprocessing import textPreprocessing
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('omw-1.4')
import wget 
wget.download("https://raw.githubusercontent.com/sholied/Dicoding-ProjectAkhir/main/dict.txt")
# Read dataset
read_data = pd.read_csv('bbc-text.csv')
# Get dummies data
category = pd.get_dummies(read_data.category)
df_baru = pd.concat([read_data, category], axis=1)
df_baru = df_baru.drop(columns='category')
# define label
label_news = df_baru.drop('text', axis=1).values
# preprocessing text 
df=df_baru.text.apply(lambda x: textPreprocessing(x).contractions())
# clean tag etc
df=df.apply(lambda x: textPreprocessing(x).cleanTags())
# tokenization
df=df.apply(lambda X: nltk.word_tokenize(X))
# Stopwords
df=df.apply(lambda x: textPreprocessing(x).remove_stopwords())
# lemmatization
df=df.apply(lambda x: textPreprocessing(x).lemmatization())
# Steamming
df=df.apply(lambda x: textPreprocessing(x).stemming())

# Tokenizer define
maxlen = 200
content = df.values
tokenizer = Tokenizer(num_words=5000, oov_token='-')
tokenizer.fit_on_texts(content)
sekuens = tokenizer.texts_to_sequences(content)
paddedset = pad_sequences(sekuens, maxlen=maxlen)
vocab_size = len(tokenizer.word_index)+1

# split data into data train, validation (80 : 20)
X_train, X_val, y_train, y_val = train_test_split(paddedset, label_news, test_size=0.2, random_state=42)
n, num_class = y_train.shape

def plot_history(history, name_plot):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("plot {}".format(name_plot))


if __name__=="__main__":
    # callback definition
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath="model.h5",
                                                                monitor='val_loss',
                                                                save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                                    monitor='val_loss', 
                                                    factor=0.2, 
                                                    mode = 'min',
                                                    patience=4, 
                                                    min_lr=0.00000001)

    #define custom early stopping when reach accuracy above 98%
    class myCustomEarlyStopping(tf.keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs={}): 
            if(logs.get('val_accuracy') > 0.90):
                print("\n Stop training, the model has reached {}% val accuracy!!".format(logs.get('val_accuracy')*100))   
                self.model.stop_training = True

    early_stop = myCustomEarlyStopping()

    callbacks = [model_checkpoint_callback, reduce_lr, [early_stop]]

    model = modelLSTM(vocab_size, num_class).lstmModel()
    num_epochs = 30
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=10, 
                        validation_data=(X_val, y_val), verbose=1, callbacks = callbacks)

    mymodel = tf.keras.models.load_model("model.h5")
    loss, accuracy = mymodel.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = mymodel.evaluate(X_val, y_val, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history, "history accuracy")