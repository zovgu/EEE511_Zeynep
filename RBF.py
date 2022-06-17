import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


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
rbf = GaussianProcessClassifier(kernel=0.0000001 * RBF(0.0000001), random_state=1)
rbf.fit(Xtrain,Ytrain)
YpredRBF = rbf.predict(Xtest)

print("Classification Report RBF")
crRBF = classification_report(Ytest,YpredRBF)
print(crRBF)

print("Confusion Matrix RBF")
ConfMatRBF = confusion_matrix(Ytest, YpredRBF)
print(ConfMatRBF)

CrossValRBF = KFold(n_splits=10, random_state=1, shuffle=True)
CVScoresRBF = cross_val_score(rbf, Xtest, Ytest, scoring='accuracy', cv = CrossValRBF, n_jobs=-1)
print("Cross Validation RBF")
print(CVScoresRBF)
