
from .lib_imports import * 

def accuracy_score(ypred, y_test):
	return (np.sum(ypred == y_test)/len(y_test))