"""
==============
Non-linear SVM
==============

Perform binary classification using non-linear SVC
with RBF kernel. The target to predict is a XOR of the
inputs.

The color map illustrates the decision function learned by the SVC.
"""
#print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


#df = pd.read_csv("C:/Users/red/Downloads/WinPython-64bit-3.6.3.0Qt5/MachineLearning/GB_metal_dataset/RGBTexXY_comma.csv")
#df = pd.read_csv("C:/Users/red/Downloads/WinPython-64bit-3.6.3.0Qt5/MachineLearning/Cast_Wt_crowsnest/initial/cast_wrought_csv_minus1.csv")
df = pd.read_csv("C:/Users/red/Downloads/WinPython-64bit-3.6.3.0Qt5/MachineLearning/Cast_Wt_crowsnest/2-27-18/RGBTexXY_collage_test_thresh80csv.csv")    
Xdf = df.iloc[:,1:14]
#Xdf = df.iloc[:,1:4]
ydf = df.iloc[:,0]
X = Xdf.values
y = ydf.values

testdf = pd.read_csv("C:/Users/red/Downloads/WinPython-64bit-3.6.3.0Qt5/MachineLearning/Cast_Wt_crowsnest/2-27-18/RGBTexXY_collage_train_thresh80csv.csv")    
testXdf = testdf.iloc[:,1:14]
testydf = testdf.iloc[:,0]
testX = testXdf.values
testy = testydf.values

scaler = StandardScaler()
X = scaler.fit_transform(X)
testX = scaler.fit_transform(testX)

#xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     #np.linspace(-3, 3, 500))
##np.random.seed(0)
##X = np.random.randn(300, 2)
##Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# fit the model
##clf = svm.NuSVC()
##clf.fit(X, y)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(X, y)

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()


## Test the svm
clf = svm.SVC(kernel='rbf',C= grid.best_params_['C'], gamma = grid.best_params_['gamma'])
clf.fit(X, y)
print(clf.predict(testX))
print(testy)
print(clf.score(testX,testy))
print(clf.intercept)


# plot the decision function for each datapoint on the grid
##Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
##Z = Z.reshape(Xdf.shape)
##
##plt.imshow(Z, interpolation='nearest',
##           extent=(Xdf.min(), Xdf.max(), ydf.min(), ydf.max()), aspect='auto',
##           origin='lower', cmap=plt.cm.PuOr_r)
##contours = plt.contour(Xdf, ydf, Z, levels=[0], linewidths=2,
##                       linetypes='--')
##plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
##            edgecolors='k')
##plt.xticks(())
##plt.yticks(())
###plt.axis([-3, 3, -3, 3])
##plt.show()
