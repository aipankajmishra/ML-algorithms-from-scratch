from .lib_imports import *


def most_common_element(lst):
	return max(set(lst),key = lst.count)

def euclidean_distance(test_instance, train_instance):
	_sum = np.sum([(val - train_instance[idx])**2 for idx,val in enumerate(test_instance)])
	return np.sqrt(_sum)