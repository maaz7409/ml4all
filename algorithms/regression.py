import numpy as np
from time import time,sleep
import matplotlib.pyplot as plt

# Conventions : 


# Explanation of numpy stuffs like broadcasting

class LinearRegression :
    def __init__(self,Weight,bias):
        # making private attributes, so they cant be directly accessed by users
        self.__W = Weight
        self.__b = bias
        self.__cost_function_history = []

    def __model(self,X) :
        return np.matmul(X,self.__W.T).T + (self.__b) # this can be reduced to np.matmul(W,X.T) using (X(W)^T)^T = (W^T)^T X^T = WX^T , where A^T denotes transpose of matrix
        # We used matrix multiplication instead of using dot product of vectors to avoid using for loops on each training set
        # <Detailed Explanation insert it here>
        # Adding a scalar to matrix adds the scalar to each element of matrix

    def __cost_function(self) : # even tho cost function is function of w & b, we dont really need it here (as w & b are taken as parameters through class)
        return np.sum(1/(2*self.__m) * (self.__model(self.__X) - self.__Y)**2 )
        # we have used vectorization here to ensure faster & parallel calculations
        # we can use in-built sum() as well but np.sum() is faster for type numpy.ndarray 



    def __gradient_descent(self,steps,learning_rate,pause_duration=0) : 
        self.a = learning_rate
        self.steps = steps
        self.pause_duration = pause_duration

        for i in range(steps) :
            tempW = self.__W - self.a/self.__m*np.matmul((self.__model(self.__X ) - self.__Y),self.__X)  #this part is art of vectorization , write the gradient descent equation for w_j^(i) and observe the terms to convert them into matrix multiplication
            tempb = self.__b - self.a/self.__m*np.sum((self.__model(self.__X) - self.__Y))
            self.__W = tempW
            self.__b = tempb
            # we used temporary variables for simultaneous update of W & b 
            # tho, we could have done that in one line using W,b ( but that'd be too long line & unreadable)
            
            print(f"({i+1}/{steps}) Cost function value : {self.__cost_function()}")
            self.__cost_function_history.append(self.__cost_function())
            
            # prints cost function after every iteration to see if it is really decreasing/converging or not
            sleep(pause_duration)

    def scale_features(self,X,mode='Zscore') :
        # In numpy.method(...,axis=,...), 
        # argument axis=0 means change is happened in rows ( like no. of rows changed ) i.e. operation is performed down the column, so as to change number of rows, resulting 1D array
        # argument axis=1 means change is happened in columns ( like no. of columns changed ) i.e. operation is performed along the rows, so as to change number of columns, resulting 1D array
        # Note that in above two cases, output is 1D array and not matrix. We will shape it to matrix using .reshape()
        if mode == 'Max-Abs' :
            return X / np.abs(np.max(X,axis=1).reshape((X.shape[0],1)))
        if mode == 'Mean' :
            return (X - np.mean(X,axis=1).reshape((X.shape[0],1)))/(np.max(X,axis=1) - np.min(X,axis=1)).reshape((X.shape[0],1))
        if mode == 'Zscore' :
            return (X - np.mean(X,axis=1).reshape((X.shape[0],1)))/(np.std(X,axis=1)).reshape((X.shape[0],1))
        
        #Explanation for using .reshape() :


        #numpy provides various methods for vectorized mathematical operations. Such methods like .max(), .mean() are used above

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
        Yhat = self.__model(Xtest) # Yhat is prediction matrix, containing all predicted values by a model
        ape = np.round(np.abs(Yhat-Ytest)/np.abs(Ytest),6)*100 # Ytest is target matrix, containing all target values & APE is Absolute Percentage Errors
        for j in range(Xtest.shape[0]) : # this gives no. of rows in Xtest matrix, which is nothing but no. of training examples
            print(f"Test case : ({j+1}/{Xtest.shape[0]}) Predicted/Target : ({Yhat[0][j]}/{Ytest[0][j]}) Absolute Percentage Error : {ape[0][j]} %")
            sleep(pause_duration)

        mae = 1/Ytest.shape[1]*np.mean(np.abs(Yhat-Ytest),axis=1)
        mse = 1/Ytest.shape[1]*np.mean((Yhat-Ytest)**2,axis=1)
        rmse = (1/Ytest.shape[1]*np.mean((Yhat-Ytest)**2,axis=1))**0.5
        smape = np.mean(np.abs(Yhat-Ytest)/(np.abs(Yhat) + np.abs(Ytest)),axis=1)*2*100
        r_squared = 1 - (np.sum((Yhat - Ytest)**2))/(np.sum((Yhat - np.mean(Ytest))**2))

        print(f"Finished testing, with \nMean Absolute Percentage Error : {np.mean(ape,axis=1)} % \nMedian Absolute Percentage Error : {np.median(ape,axis=1)} % \nMean Absolute Error : {mae} \nMean Squared Error : {mse} \nRMSE : {rmse} \nSymmetric Mean Absolute Percentage Error : {smape} %\nR^2 value : {r_squared}")


    def predict(self,Xtest) :
        print("Predicting values... \n")
        return self.__model(Xtest)

