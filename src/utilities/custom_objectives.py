import theano.tensor as T
from theano import function

def frob_norm(y_true, y_pred):
	y = T.dmatrix('y')
	y_hat = T.dmatrix('y_hat')

	diff = y - y_hat
	conj_trans_diff = diff.T
	A = T.dot(conj_trans_diff, diff)
	frob_norm = T.sqrt(A.trace())

	f = function([y, y_hat], frob_norm)
	return f