from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate damped sine wave in [0,1]
def generate_sequence(length, period, decay):
	return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]

# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
	X, y = list(), list()
	for _ in range(n_patterns):
		p = randint(10, 20)
		d = uniform(0.01, 0.1)
		sequence = generate_sequence(length + output, p, d)
		X.append(sequence[:-output])
		y.append(sequence[-output:])
	X = array(X).reshape(n_patterns, length, 1)
	y = array(y).reshape(n_patterns, output)
	return X, y

# configure problem
length = 50
output = 5

# define model
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(length, 1)))
model.add(LSTM(20,return_sequences=True))  #<------Middle stack
model.add(LSTM(20))

model.add(Dense(output))
model.compile(loss='mae', optimizer='adam',metrics=['acc'])
model.summary()

# fit model
X, y = generate_examples(length, 10000, output)
print(X)
print(X.shape)
print("------------y======")
print(y)
print(y.shape)

print("=============Fit Model===============")
history = model.fit(X, y, batch_size=500, epochs=1)

# evaluate model
X, y = generate_examples(length, 1000, output)
loss = model.evaluate(X, y, verbose=0)
print(loss)

# prediction on new data
X, y = generate_examples(length, 1, output)
print("=============Prediction")
print(X)
print(y)
yhat = model.predict(X, verbose=0)
pyplot.plot(y[0], label='yInput')
pyplot.plot(yhat[0], label='yPredict')
pyplot.legend()
pyplot.show()