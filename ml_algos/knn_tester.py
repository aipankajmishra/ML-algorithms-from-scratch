import sys
sys.path.append('../')

from lib_imports import *

def main():
	print("Running the KNN algorithm implementation from scratch")

	# Loading the dataset, we will use KNN for solving a classification problem

	X, y = load_iris(return_X_y = True)

	# Now dividing the dataset we got into the train and test
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 42)

	"""
		Now we have the split, lets call the KNN implementation to fit the train data and have the predictions
	"""

	clf = KNN(k = 10)

	clf.fit(X_train,y_train)

	ypred = clf.predict(X_test)

	print("The predictions are - ")
	print(ypred)

	print("The accuracy is - ")
	accuracy = accuracy_score(ypred,y_test)
	print(accuracy)


if __name__ == "__main__":
	main()