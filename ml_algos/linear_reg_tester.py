

import sys
sys.path.append('../')


from lib_imports import *



def main():
	print("Running the linear regression algorithm from scratch")
	print("-----------------------")
	print("| Linear regression   |")
	print("-----------------------")

	X, y = load_boston(return_X_y = True)
	print(X.shape,y.shape)





if __name__ == "__main__":
	main()