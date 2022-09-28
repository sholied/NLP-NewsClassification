import tensorflow as tf

# create model LSTM
class modelLSTM():
    
    def __init__(self, inputDim, numClass):
        self.inputDim = inputDim
        self.numClass = numClass

    def lstmModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.inputDim, output_dim=50, input_length=200),
            tf.keras.layers.SpatialDropout1D(0.15),
            tf.keras.layers.LSTM(50, dropout=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.numClass, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        print(model.summary())
        return model