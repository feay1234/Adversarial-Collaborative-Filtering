from keras.layers import Input, Embedding, dot, Add, Flatten, Lambda, Dense;
from keras.models import Model;

class MatrixFactorization:
    def __init__(self, uNum, iNum, dim):
        # Define user input -- user index (an integer)
        userInput = Input(shape=(1,), dtype="int32")
        itemInput = Input(shape=(1,), dtype="int32")
        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim)
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim)

        uEmb = Flatten()(userEmbeddingLayer(userInput))
        iEmb = Flatten()(itemEmbeddingLayer(itemInput))

        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])


