from .lib_imports import *

class KNN:

	def __init__(self,k = 3):
		self.k = k 

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y 


	def label_assign(self, closest_k):
		common_label = most_common_element(self.y_train[closest_k].tolist())
		return common_label

	def _predict(self, x):
		"""
			This method returns output for a single instace.
			We will call this method for all instance in X_test
			
			Calculate distance of this instance from all other in the training set. 
			After calculating the distance, we will select the closest k from the given sample.

			After that, we will vote and get the value which is in the majority.
		"""

		distances = []

		for i, train_record in enumerate(self.X_train):
			eucl_distance =  euclidean_distance(x,train_record)
			distances.append(eucl_distance)

		""" 
		Now we have calculated the euclidean distance of the instance from each  sample in the train, 
		we will try labelling it using the closes k neighbor's label. 
		"""
		closest_k = np.argsort(distances)[:self.k]
		
		assign_label = self.label_assign(closest_k)
		return assign_label


	def predict(self, X_test):
		return np.asarray([self._predict(x) for x in X_test])

