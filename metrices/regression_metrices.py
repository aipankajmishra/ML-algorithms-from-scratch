

from .lib_imports import *


# Calculating the rmse from true and predicted values.
def rmse_score(ypred, ytrue):
	mse = np.sum((ypred - ytrue)**2)
	_len = len(ypred)
	return np.sqrt(mse/_len)


