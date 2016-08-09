#Prototype: http://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
# load the data
import pandas as pd
# Read Models data
my_data = pd.read_csv('smartcabModelsComparison.csv')
print "Models data read successfully!"

# all columns but last are features:
feature_cols = list(my_data.columns[:-1])
target_col = my_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

# feature values for all students:
X_all = my_data[feature_cols]
# corresponding targets/labels: 
y_all = my_data[target_col]#.replace(['yes', 'no'], [1, 0])
print "\nFeature values:-"

# prepare a range of alpha values to test
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X_all, y_all) #grid.fit(dataset.data, dataset.target)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)
print(grid.best_estimator_)


#Note: The grid.best_score_ is the positive version of what shown below, Details down.


#Outcome
# Andreas-MacBook-Pro:smartcab andreamelendezcuesta$ python kindaGridSearchModel.py
# Models data read successfully!
# Feature column(s):-
# ['ON_TIME_ARRIVALS']
# Target column: MODEL

# Feature values:-
# GridSearchCV(cv=None, error_score='raise',
#        estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#    normalize=False, random_state=None, solver='auto', tol=0.001),
#        fit_params={}, iid=True, n_jobs=1,
#        param_grid={'alpha': array([  1.00000e+00,   1.00000e-01,   1.00000e-02,   1.00000e-03,
#          1.00000e-04,   0.00000e+00])},
#        pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)
# -11.2696551016
# 1.0
# Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#   normalize=False, random_state=None, solver='auto', tol=0.001)



# Explanation: 

# Trying to close this out, so am providing the answer that David and larsmans have eloquently described in the comments section:

# Yes, this is supposed to happen. The actual MSE is simply the positive version of the number you're getting.

# The unified scoring API always maximizes the score, so scores which need to be minimized are negated in order for the unified scoring API to work correctly. The score that is returned is therefore negated when it is a score that should be minimized and left positive if it is a score that should be maximized.

# This is also described in sklearn GridSearchCV with Pipeline.


# Details: http://stackoverflow.com/questions/21443865/scikit-learn-cross-validation-negative-values-with-mean-squared-error




#Prototype: http://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/

# import numpy as np
# from sklearn import linear_model
# from sklearn.model_selection import GridSearchCV

# load the diabetes datasets
# dataset = datasets.load_diabetes()

# # prepare a range of alpha values to test
# alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# # create and fit a ridge regression model, testing each alpha
# model = linear_model.Ridge()
# grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
# grid.fit(dataset.data, dataset.target)
# print(grid)
# # summarize the results of the grid search
# print(grid.best_score_)
# print(grid.best_estimator_.alpha)


