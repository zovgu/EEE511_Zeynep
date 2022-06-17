import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import sklearn
from numpy import unique
from numpy import where
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report




 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
         
        print ("center"), self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
        print (G)
         
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

# read the dataset with the attributes as the title Optimal depth of win = ODOW
dataset = pd.read_csv("krkopt.data", names = ["White King File","White King Rank", "White Rook File", "White Rook Rank", "Black King File", "Black King Rank", "ODOW" ])
data = dataset.values

#separate input and output
X = data[:,:-1].astype(str)
y = data[:, -1].astype(str)


# encode files 
o_enc = OrdinalEncoder()
X_enc = o_enc.fit_transform(X)
l_enc = LabelEncoder()
y_enc = l_enc.fit_transform(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_enc,y_enc, test_size=0.3, shuffle=True, random_state=1)
 
# RBF
print("RBF with sklearn")
rbf = RBF(6,200,6)
rbf.train(Xtrain,Ytrain)
YpredRBF = rbf.test(Xtest)
 
YpredRBF = np.int64((YpredRBF>0.5)*1) #for consistency

print("Classification Report RBF")
crRBF = classification_report(Ytest,YpredRBF)
print(crRBF)
