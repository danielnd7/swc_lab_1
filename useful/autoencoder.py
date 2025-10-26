# lstm autoencoder recreate sequence
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Input

# define input sequence
sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_samples = 1
n_in = len(sequence)
n_features = 1
sequence = sequence.reshape((n_samples, n_in, n_features))

# define model
model = Sequential()
model.add(Input(shape=(n_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)

# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat[0,:,0])

print(sequence.shape)
print(yhat.shape)
error = np.mean((sequence - yhat)**2)
print("Error cuadrático medio:", error)