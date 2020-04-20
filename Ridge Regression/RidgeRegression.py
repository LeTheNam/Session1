import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class RidgeRegression:
    def __init__(self, numFold, numBatch, lamda, learningRate, numIter):
        self.numIter      = numIter
        self.numFold      = numFold
        self.numBatch     = numBatch
        self.Lamda        = lamda 
        self.learningRate = learningRate

    def fit(self, X, y, lamda):
        N, d = X.shape # 60x16  (N x (d + 1))  N: number of instances, d : number of attributes 
        I    = np.eye(d)
        W    = np.linalg.inv(X.T@X + lamda*I) @ X.T @ y 
        return W

    def fitMiniBatchGradientDescent(self, X, y, lamda):
        learningRate = 0.005
        N, d         = X.shape                 # X: N x d
        W            = np.random.rand(d,1)    # W: d x 1 
        batch        = np.linspace(0, N, num= self.numBatch+1, dtype= int)
        for i in range(self.numIter):  
            randomId = np.random.permutation(N)
            X        = X[randomId]
            y        = y[randomId]
            for j in range(self.numBatch):
                X_train = X[batch[j]:batch[j+1]]
                y_train = y[batch[j]:batch[j+1]]                
                y_pred  = X_train @ W                                   
                grad    = X_train.T @ (y_pred - y_train) + lamda * W                 
                W      -= self.learningRate * grad
        return W

    def predict(self, weight, X):
        dim = X.shape
        return (X@weight)

    def computeLossRSS(self, y_pred, y):
        N = y.shape[0]
        r = y_pred - y 
        return 1/N*np.sum(r*r)

    def KFoldCrossValidation(self, X, y):
        N, d    = X.shape
        minLoss = np.inf
        lamda   = 0
        lenFold = np.int(N/self.numFold)
        fold    = np.linspace(0, N, num= self.numFold+1, dtype= int)
        W_best  = np.zeros([d, 1])
        for ld in self.Lamda:
            aver = 0.
            for i in range(self.numFold):
                X_train = np.concatenate((X[:fold[i]], X[fold[i+1]:N]))
                y_train = np.concatenate((y[:fold[i]], y[fold[i+1]:N]))
                # W       = self.fit(X_train, y_train, ld)
                W       = self.fitMiniBatchGradientDescent(X= X_train, y= y_train, lamda= ld)
                loss    = self.computeLossRSS(X[fold[i]:fold[i+1]] @ W, y[fold[i]:fold[i+1]])
                aver   += loss
            aver = aver/self.numFold
            if (minLoss > aver):
                minLoss = aver
                lamda   = ld
                W_best  = W
        return lamda, minLoss, W_best

def normalizeData(data):
    # Min-max scaling
    N, d  = data.shape
    x_min = np.amax(data[:, :d-1], axis= 0)
    x_max = np.amin(data[:, :d-1], axis= 0)
    # Normalize X
    X     = (data[:, :d-1] - x_min) / (x_max - x_min) 
    # build a matrix of attributes
    x0    = np.ones((N, 1))
    X     = np.hstack((x0, X))
    y     = data[:, d-1].reshape(-1, 1)
    return X, y
   
def main():
    data = np.loadtxt('death_rate_data.txt')
    dim  = data.shape
    X, y = normalizeData(data)
    X_train, X_test = X[:50], X[50:]
    y_train, y_test = y[:50], y[50:]


    # RidgeRegression with paras: numFold, numBatch, lamda, learningRate, numIter
    # Parameter
    lamda        = np.linspace(0.001, 1., num=1000)
    numIter      = 100
    learningRate = 0.01
    print(lamda)
    
    model = RidgeRegression(numFold= 10, numBatch= 4, lamda= lamda, learningRate= learningRate, numIter= numIter)
    bestLamda, minLoss, bestW = model.KFoldCrossValidation(X_train, y_train)
    print("Best lamda: {} and Min loss: {}".format(bestLamda, minLoss))
    # Train lai model voi bestLamda
    W = model.fitMiniBatchGradientDescent(X_train, y_train, bestLamda)
    print("Predicting value: ")
    print(model.predict(W, X_test))


    # Use sklearn package
    print("Use sklearn package:")
    ridge = Ridge(alpha= 0.1)
    ridge.fit(X_train, y_train)
    print(ridge.predict(X_test))

if __name__ == "__main__":
    main()