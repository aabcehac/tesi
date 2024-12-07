from test_sqlite import *

import keras
import pandas
import numpy


num_items = len(ultimate_data_dict['id'])
ultimate_data_list = [*ultimate_data_dict.values()]
X = ultimate_data_dict
y = X.pop('decile_score')
#X = [x for x in X.values()]
X = list(map(list, zip(*X.values()))) # Transpose the list
# unpacker = lambda item: tuple(*item)
Z = {tuple(X[i]): y[i] for i in range(len(y))}
# generator = ((x, y) for x, y in Z.items())

#print(type(y))
#print(len(X))

#obiettivo: fornire dati in batch al modello
batch_size=32

'''
def data_generator(X, y, batch_size):

    #numero campioni?
    num_samples=len(y)

    #fonte doc: il generatore deve produrre batch continuamente. Ipotizzo un while(true) per risolvere
    while True:
        #iterazione sui campioni in blocchi: si attraversano i dati in passi di dimensione batch_size
        for i in range(0, num_samples, batch_size):
            #estrazione dei batch
            X_batch = [x[i:i + batch_size]for x in X] 
            y_batch = y[i:i + batch_size]

            #traspongo
            X_batch = [[row[j] for row in X_batch] for j in range(len(X_batch[0]))]

            yield (X_batch, y_batch)

train_generator = data_generator(X,y,batch_size)
'''

# Generico generatore: gen_gen = (f(i) for i in iteratore_di_partenza)
# Puoi definire qualsiasi f esternamente, che trasformi un singolo elemento che l'espressione possa richiamare


                


# X = list(map(list, zip(*X.values()))) # Transpose the list
'''
model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1))

# Note that you can also omit the initial `Input`.
# In that case the model doesn't have any weights until the first call
# to a training/evaluation method (since it isn't yet built):
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
# model.weights not created yet

# Whereas if you specify an `Input`, the model gets built
# continuously as you are adding layers:
model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
len(model.weights)  # Returns "2"

# When using the delayed-build pattern (no input shape specified), you can
# choose to manually build your model by calling
# `build(batch_input_shape)`:
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
model.build((None, 16))
len(model.weights)  # Returns "4"

# Note that when using the delayed-build pattern (no input shape specified),
# the model gets built the first time you call `fit`, `eval`, or `predict`,
# or the first time you call the model on some input data.
'''
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1))
model.compile(optimizer='sgd', loss='mse')

#steps_per_batch=len(y)
model.fit(Z)
# This builds the model for the first time:
#model.fit(X, y, batch_size=32, epochs=10)
print(type(y))
print(len(X))
