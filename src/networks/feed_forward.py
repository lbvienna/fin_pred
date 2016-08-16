import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import backend as K

from utilities import custom_objectives

def model(output_dim=90):
	input_shape = X.shape[0]

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
		loss = custom_objectives.frob_norm
	model.compile(optimizer=adam, loss='mse')

	return model
