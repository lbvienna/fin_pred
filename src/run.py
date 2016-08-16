from sklearn.cross_validation import train_test_split
from utilities import utils
from networks import feed_forward

def main():
	X, y = utils.read_data()
	k_fold(X, y)

def run_network(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=16):
	model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size)
	score = model.evaluate(X_test, y_test, batch_size=batch_size)
	print "		evaluation_metric: {0}".format(score)
	return score

def k_fold(X, y, k_folds=10, seed=1234):
	scores = []
	for k in xrange(k_folds):
		model = feed_forward.model()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
		score = run_network(model, X_train, X_test, y_train, y_test)
		scores.append(score)
	print "average over {0}-folds: {1}".format(k_folds, np.mean(scores))