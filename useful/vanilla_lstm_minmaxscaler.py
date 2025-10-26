# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# split a univariate sequence into samples
def split_sequence(sequence, sequence_scaled,n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence_scaled[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Normalizar los datos
scaler = MinMaxScaler() # Usar esta función
scaler_array = np.array(raw_seq).reshape(-1, 1)
raw_seq_scaled = scaler.fit_transform(scaler_array)

print(raw_seq)
print(raw_seq_scaled)

input("raw_sew vs raw_seq_scaled")

# choose a number of time steps
n_steps = 3
# split into samples
#X, y = split_sequence(raw_seq,raw_seq_scaled, n_steps)

windows = np.lib.stride_tricks.sliding_window_view(raw_seq_scaled.reshape(-1), window_shape=(n_steps))
X = array(windows[:-1])  # Todas las ventanas menos la última. Hay que convertir a array numpy
y = array(raw_seq[n_steps:])  # Lo que predices (el siguiente valor después de cada ventana)

print(X[0])
print(y[0])
input('X[0] y[0]')

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Input(shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=200)

# demonstrate prediction
x_input = array([70, 80, 90])
x_input = np.array(x_input).reshape(-1, 1)

x_input_scaled = scaler.transform(x_input)
print(x_input_scaled)
input('x_input_scaled to predict')

x_input_scaled = x_input_scaled.reshape((1, n_steps, n_features))
yhat = model.predict(x_input_scaled, verbose=0)
print(yhat)