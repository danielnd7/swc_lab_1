# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
import numpy as np

# split a multivariate sequence into samples
def split_sequences(input_sequence, output_sequence, n_steps):
	X, y = list(), list()
	for i in range(len(input_sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(input_sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = input_sequence[i:end_ix], output_sequence[end_ix-1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
input_seq= hstack((in_seq1, in_seq2)) # row,column format
output_seq = out_seq.reshape((len(out_seq), 1))

print(input_seq)
print(output_seq)

n_steps = 3
n_features = 2

# convert into input window / next output
#X, y = split_sequences(input_seq,output_seq, n_steps)

X = np.lib.stride_tricks.sliding_window_view(hstack((in_seq1, in_seq2)), window_shape= (n_steps,n_features))
X = np.squeeze(X, axis=1) # Para eliminar eje y que no aparezca (7,1,3,2) sino (7,3,2)
y = out_seq[n_steps-1:]  # Lo que predices (el siguiente valor después de cada ventana)

print(X.shape)
print(y.shape)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(Input(shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=200)

# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
