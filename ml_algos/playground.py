
import sys

sys.path.append('../')

from lib_imports import *
from ml_algorithms.utils.general import standardize



X, y = load_boston(return_X_y = True)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 42, stratify = True)

X_train_std  = standardize(X_train)
print(X_train_std)
#y_train_std = standardize(y_train)