""" FILE USES PERCEPTRON"""

from numpy import exp
from random import random

class Perceptron(object):
	"""Class to implement perceptron"""
	
	def __init__(self,size=3):
		#This function will only initialise my weights,		
		self.weights=[]
		
		for i in range(size):
			self.weights.append(random())


	def __sigmoid(self,x):
		return (1/(1+exp(-x)))

	def __sigmoid_derivative(self,x):
		return (self.__sigmoid(x)*(1-self.__sigmoid(x)))

	def train(self,x_train,y_train,pract_time,learn_rate):
		
		for i in range(pract_time):

			for j in range(len(x_train)):
				predicted_output = self.predict(x_train[j])
				error = y_train[j]-	predicted_output
				for k in range(len(x_train[j])):
					adjust = learn_rate*x_train[j][k]*error*self.__sigmoid_derivative(predicted_output)
					self.weights[k]+=adjust


	def predict(self,input_x):
		sum_x=0.0
		for i in range(len(input_x)):
			sum_x+= input_x[i]*self.weights[i]
		return self.__sigmoid(sum_x)



if __name__ == '__main__':

    training_input = [[0,0,0],[1,0,0],[0,0,1],[0,1,1],
    					[1,0,1],[1,1,0],[0,1,0]]
    training_outputs = [0,1,0,0,1,1,0]

    nw=Perceptron()
    
    print("Printing Initial weights")
    print(nw.weights)

    nw.train(training_input,training_outputs,10000,1)
    print("Printing new weights")
    print(nw.weights)

    print("Prediction for [1, 0, 1]")
    temp_predicted = nw.predict([1,0,1])
    print(temp_predicted)
    print("Accuracy:" ,(temp_predicted)*100,"%")

