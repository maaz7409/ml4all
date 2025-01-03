import numpy as np
from time import time,sleep
import matplotlib.pyplot as plt

# Conventions : 


# Explanation of numpy stuffs like broadcasting

class LogisticRegression :
    def __init__(self,Weight,bias,threshold,activation='Sigmoid',optimization='BGD'):
        # making private attributes, so they cant be directly accessed by users
        self.__W = Weight
        self.__b = bias
        self.__threshold = threshold
        self.__activation = activation
        self.__optimization = optimization
        self.__cost_function_history = []


    def __sigmoid(self,Z) :
        return np.reciprocal(1+np.exp(-1*Z))
    

    def __model(self,X) :
        return self.__sigmoid(np.matmul(self.__W,X.T) + self.__b)
    

    def __logistic_loss_function(self,X,Y) : # we will be writing this logistic loss function in matrix form
        return -1*(np.matmul(Y,np.log(self.__model(X)).T) + np.matmul(1-Y,np.log(1-self.__model(X)).T))


    def __cost_function(self) : # even tho cost function is function of W & b, we dont really need it here (as W & b are taken as parameters through class)
        return np.mean(self.__logistic_loss_function(self.__X,self.__Y))



    def __gradient_descent(self,steps,learning_rate,pause_duration=0) : # gradient descent of logistic regression is simliar to linear regression
        self.a = learning_rate
        self.steps = steps
        self.pause_duration = pause_duration

        for i in range(steps) :
            tempW = self.__W - self.a/self.__m*np.matmul((self.__model(self.__X ) - self.__Y),self.__X)  # We write the gradient descent equation for w_j^(i) and observe the terms to convert them into matrix multiplication
            tempb = self.__b - self.a/self.__m*np.sum((self.__model(self.__X) - self.__Y))
            self.__W = tempW
            self.__b = tempb
            
            print(f"({i+1}/{steps}) Cost function value : {self.__cost_function()}")
            self.__cost_function_history.append(self.__cost_function())
            # prints cost function after every iteration to see if it is really decreasing/converging or not

            sleep(pause_duration)

    def scale_features(self,X,mode='Zscore') :

        if mode == 'Max-Abs' :
            return X / np.abs(np.max(X,axis=1).reshape((X.shape[0],1)))
        if mode == 'Mean' :
            return (X - np.mean(X,axis=1).reshape((X.shape[0],1)))/(np.max(X,axis=1) - np.min(X,axis=1)).reshape((X.shape[0],1))
        if mode == 'Zscore' :
            return (X - np.mean(X,axis=1).reshape((X.shape[0],1)))/(np.std(X,axis=1)).reshape((X.shape[0],1))
        

    def train(self,X,Y,learning_rate,steps,pause_duration=0,visual=False) :
        print("\nTraining data ....\n")
        t1 = time()
        self.__m = X.shape[0] # Gives number of rows in feature matrix as .shape() method gives (rows,columns) as output for 2D matrix  & [0] gives first element of tuple
        self.__X = X
        self.__Y = Y
        self.__gradient_descent(learning_rate=learning_rate,steps=steps,pause_duration=pause_duration)

        if visual :
            plt.title("Learning Curve")
            plt.xlabel('No. of iterations')
            plt.ylabel('Cost Function value')
            plt.plot(list(range(1,self.steps+1)),self.__cost_function_history)

        t2 = time()
        print(f"Finished training Data. Time taken for training : {t2-t1} seconds\n(including {self.pause_duration*self.steps} second(s) of pause).\n")
    
    def test(self,Xtest,Ytest,pause_duration=0) :
        print("\nTesting Data....\n")
        Yhat = self.__model(Xtest) > self.__threshold # Yhat is prediction matrix, this will make matrix of Trues and False, depending on if an element > threshold or not
        corrects = Yhat == Ytest
        incorrects = Yhat != Ytest

        for j in range(Xtest.shape[0]) : # this gives no. of rows in Xtest matrix, which is nothing but no. of training examples
            print(f"Test case : ({j+1}/{Xtest.shape[0]}) Predicted/Target : ({Yhat[0][j]}/{Ytest[0][j]}) Prediction : {'Correct' if corrects[0][j] else 'Incorrect'}")
            sleep(pause_duration)


        print(f"Finished Testing, With \nTotal Correct Predictions : {np.sum(corrects)} \nTotal Incorrect Predictions : {np.sum(incorrects)} \nAccuracy = {corrects/Xtest.shape[0]*100}")
        
        


    def predict(self,Xtest) :
        print("Predicting values... \n")
        return np.int_(self.__model(Xtest) > self.__threshold) # np.int_() will convert all the boolean values to integers 1 and 0

