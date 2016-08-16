import numpy as np

from sklearn.cross_validation import train_test_split

import theano.tensor as T
from theano import function

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import backend as K

def main():
	feed_forward()

def frob_norm(y_true, y_pred):
	y = T.dmatrix('y')
	y_hat = T.dmatrix('y_hat')

	diff = y - y_hat
	conj_trans_diff = diff.T
	A = T.dot(conj_trans_diff, diff)
	frob_norm = T.sqrt(A.trace())

	f = function([y, y_hat], frob_norm)
	return f

def feed_forward(X, y, k_folds=10, output_dim=90):
	input_shape = X.shape[0]
	batch_size = 16
	seed = 1234

	model = Sequential()
	model.add(Dense(1024, init='he_uniform', batch_input_shape=(None, input_shape)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(512, init='he_uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(256, init='he_uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(128, init='he_uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(output_dim, init='he_uniform'))

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	#specify own metric function?
	loss = 'mse'
	if output_dim > 1:
		loss = frob_norm
	model.compile(optimizer=adam, loss='mse')

	for k in xrange(k_folds):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
		model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size)
		score = model.evaluate(X_test, y_test, batch_size=batch_size)
