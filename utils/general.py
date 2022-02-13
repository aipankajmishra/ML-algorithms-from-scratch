from .lib_imports import *


def most_common_element(lst):
	return max(set(lst),key = lst.count)

def euclidean_distance(test_instance, train_instance):
	_sum = np.sum([(val - train_instance[idx])**2 for idx,val in enumerate(test_instance)])
	return np.sqrt(_sum)


def standardize(X):

	X_mean = X.mean(axis = 0) # Mean operation about rows
	X_std = X.std(axis = 0) # Find Standard deviation about rows

	#X_std = (X - X_mean)/X_std, doesnt work like this in case of numpy array
	try:
		for idx in range(X.shape[1]): # Iterate on columns
			X[:idx] = (X[:idx] - X_mean[idx])/(X_std[idx])
	except:
		X = (X - X_mean)/(X_std)

	return X



