import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
sns.set(style="whitegrid")
sns.set_color_codes("pastel")

# Read our dataset
mathTransformed = pd.read_csv('Datasets/math_transformed.csv')
portTransformed = pd.read_csv('Datasets/port_transformed.csv')

# We are predicting quarter 1 grades
targetColumn = 'G1'
# We don't want to use G1, G2, or G3 to predict G1
featureColumns = [i for i in list(portTransformed) if i not in ['Walc', 'G1', 'G2', 'G3']]

def return_residuals(model, df, col_to_predict):
    '''
    Using a 90/10 split of data, returns list of residuals for predictions
    of testing set.
    '''
    all_columns = list(df)
    all_X = [i for i in all_columns if i not in ['G1', 'G2', 'G3']]
    y = df[col_to_predict]
    X = df[all_X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model.fit(X_train, y_train)
    all_residuals = []
    residuals = model.predict(X_test)
    y_test = list(y_test)
    for i in range(len(y_test)):
        residual = (y_test[i] - residuals[i])
        all_residuals.append(residual)
    return all_residuals

# def random_forests(dataset, target_col, feat_cols, NE, MD, MSS, MSL):
#     model = RandomForestRegressor()
#     parameters = {'n_estimators' : [NE],
#                   'max_depth' : [MD],
#                   'min_samples_split' : [MSS],
#                   'min_samples_leaf' : [MSL]}
#     test_scores = []
#     X = dataset[feat_cols]
#     y = dataset[target_col]
#     #for score in ['r2','neg_mean_squared_error', 'neg_mean_absolute_error']:
#     score = 'r2'
#     clf = GridSearchCV(model, parameters, cv=5, scoring=score)
#     clf.fit(X, y)
#     test_scores.append(clf.best_score_)
#     s = clf.best_estimator_.feature_importances_
#     sorted_feat_idx = sorted(range(len(s)), key=lambda k: s[k])
#     importances = [s[i] for i in reversed(sorted_feat_idx)]
#     features = [feat_cols[i] for i in reversed(sorted_feat_idx)]
#     return clf.best_estimator_, test_scores, importances[:7], features[:7]

def random_forests(dataset, target_col, feat_cols, a, b, c, d):
    model = RandomForestRegressor()
    n_features = 7
    parameters = {'n_estimators' : [a],
                  'max_depth' : [b],
                  'min_samples_split' : [c],
                  'min_samples_leaf' : [d]}
    X = dataset[feat_cols]
    y = dataset[target_col]
    # for score in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
    #     clf = GridSearchCV(model, parameters, cv=5, scoring=score)
    #     clf.fit(X, y)
    #     print(score, ':', '%0.2f' % clf.best_score_)
    count = 0
    print(feat_cols)
    while True:
        count += 1
        clf = GridSearchCV(model, parameters, cv=5, scoring='r2')
        clf.fit(X, y)
        s = clf.best_estimator_.feature_importances_
        sorted_feat_idx = sorted(range(len(s)), key=lambda k: s[k])
        importances = [s[i] for i in reversed(sorted_feat_idx)]
        features = [feat_cols[i] for i in reversed(sorted_feat_idx)]
        if (('Fedu' in features[:n_features] and 'Medu' in features[:n_features]) or count == 1000):
            print(count)
            break
    print(clf.best_score_)
    print('n_estimators:', clf.best_params_['n_estimators'])
    print('max_depth:', clf.best_params_['max_depth'])
    print('min_samples_split:', clf.best_params_['min_samples_split'])
    print('min_samples_leaf:', clf.best_params_['min_samples_leaf'])
    return clf.best_estimator_, importances[:n_features], features[:n_features]

# def random_forests(dataset, target_col, feat_cols, NE, MD, MSS, MSL):
#     model = RandomForestRegressor(NE, max_depth=MD, min_samples_split=MSS, min_samples_leaf=MSL)
#     return model

# def random_forests(dataset, target_col, feat_cols, NE, MD, MSS, MSL):
#     model = RandomForestRegressor()
#     parameters = {'n_estimators' : [NE],
#                   'max_depth' : [MD],
#                   'min_samples_split' : [MSS],
#                   'min_samples_leaf' : [MSL]}
#     X = dataset[feat_cols]
#     y = dataset[target_col]
#     clf = GridSearchCV(model, parameters, cv=5, scoring='r2')
#     clf.fit(X, y)
#     return clf.best_estimator_

def doKNN(dataset, target_col, feat_cols):
    model = KNeighborsRegressor()
    parameters = {'n_neighbors' : np.arange(30, 51),
                  'weights'     : ['uniform', 'distance'],
                  'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute']}
    X = dataset[feat_cols]
    y = dataset[target_col]
    clf = GridSearchCV(model, parameters, cv=5, scoring='r2')
    clf.fit(X,y)
    return clf.best_estimator_

def linearRegression(originalDf, targetColumn, featureColumns):
    '''
    Runs unregularized linear regression and returns the 
    R^2 and mean squared error (averaged over all folds 
    via cross validation)
    '''
    lm = linear_model.LinearRegression(normalize=True)
    X = originalDf[featureColumns]
    y = originalDf[targetColumn]
    scores = cross_validate(lm, X, y, scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'], cv=10, return_train_score=False)
    return lm, scores

def linearRegressionRidge(originalDf, targetColumn, featureColumns):
    '''
    Runs ridge regularized linear regression and returns the 
    mean and standard deviation of test scores
    '''
    parameters = {'alpha' : np.arange(0.0001, 0.01, 0.01)}
    scoringMethods = ['r2','neg_mean_squared_error', 'neg_mean_absolute_error']
    test_scores = []
    lm = linear_model.Ridge(normalize=True)
    for score in scoringMethods:
        X = originalDf[featureColumns]
        y = originalDf[targetColumn]
        clf = GridSearchCV(lm, parameters, cv=10, scoring=score)
        clf.fit(X,y)
        test_scores.append(clf.best_score_)
    return clf.best_estimator_, test_scores

def linearRegressionLasso(originalDf, targetColumn, featureColumns):
    '''
    Runs lasso regularized linear regression and returns the 
    mean and standard deviation of test scores
    '''
    parameters =  {'alpha' : np.arange(0.0001, 0.1, 0.01)}
    scoringMethods = ['r2','neg_mean_squared_error', 'neg_mean_absolute_error']
    test_scores = []
    lm = linear_model.Lasso(normalize=True)
    for score in scoringMethods:
        X = originalDf[featureColumns]
        y = originalDf[targetColumn]
        clf = GridSearchCV(lm, parameters, cv=10, scoring=score)
        clf.fit(X,y)
        test_scores.append(clf.best_score_)
    return clf.best_estimator_, test_scores

def runLinearRegression(df, targetColumn, featureColumns):
    '''
    This function runs all the linear regression models on the specific
    dataset using specified target and feature columns and returns back
    the scores as a list of tuples (a,b, c) where a corresponds to the model, 
    b corresponds to the R^2 scores and c corresponds to the mean squared errors
    (over all runs).
    '''
    unregularized_lm, unregularized = linearRegression(df, targetColumn, featureColumns)
    ridge_lm, ridge = linearRegressionRidge(df, targetColumn, featureColumns)
    lasso_lm, lasso = linearRegressionLasso(df, targetColumn, featureColumns)
    return [[unregularized_lm, unregularized['test_r2'], unregularized['test_neg_mean_squared_error'], unregularized['test_neg_mean_absolute_error']],
            [ridge_lm, ridge[0], ridge[1], ridge[2]],
            [lasso_lm, lasso[0], lasso[1], lasso[2]]]

def SVM(originalDf, targetColumn, featureColumns):
    '''
    Runs ridge regularized linear regression and returns the 
    mean and standard deviation of test scores
    '''
    parameters = {'C' : np.arange(0.1, 10, .5)}
    test_scores = []
    X = originalDf[featureColumns]
    y = np.ravel(originalDf[targetColumn])
    svr = SVR()
    #scoringMethods = ['r2','neg_mean_squared_error', 'neg_mean_absolute_error']
    scoringMethods = ['neg_mean_absolute_error']
    for score in scoringMethods:
        clf = GridSearchCV(svr, parameters, cv=10, scoring=score)
        clf.fit(X,y)
        test_scores.append(clf.best_score_)
    return clf.best_estimator_, test_scores

#print('Math:')
#best_model, test_scores, imp, feat = random_forests(mathTransformed, targetColumn, featureColumns, 46, 3, 4, 2)
# print("r2:  %0.2f" % (test_scores[0]))
# # print("MSE: %0.2f" % (test_scores[1]))
# # print("MAE: %0.2f" % (test_scores[2]))
# math_resid = return_residuals(best_model, mathTransformed, targetColumn)

print('Portuguese:')
best_model, imp, feat = random_forests(portTransformed, targetColumn, featureColumns, 42, 9, 4, 1)
# print("r2:  %0.2f" % (test_scores[0]))
# # print("MSE: %0.2f" % (test_scores[1]))
# # print("MAE: %0.2f" % (test_scores[2]))
# port_resid = return_residuals(best_model, portTransformed, targetColumn)

# Get linear regression residuals
# unregularized, ridge, lasso = runLinearRegression(portTransformed, targetColumn, featureColumns)
# port_lm_resid = return_residuals(lasso[0], portTransformed, 'G1')
# unregularized, ridge, lasso = runLinearRegression(mathTransformed, targetColumn, featureColumns)
# math_lm_resid = return_residuals(lasso[0], mathTransformed, 'G1')
# print('Done with Linear Regression')

# # Get SVM residuals
# port_svm_best, results = SVM(portTransformed, targetColumn, featureColumns)
# port_svm_resid = return_residuals(port_svm_best, portTransformed, 'G1')
# math_svm_best, results_math_svm = SVM(mathTransformed, targetColumn, featureColumns)
# math_svm_resid = return_residuals(math_svm_best, mathTransformed, 'G1')
# print('Done with SVM')

# # Get tree residuals
# port_tree = random_forests(portTransformed, targetColumn, featureColumns, 43, 9, 3, 2)
# port_tree_resid = return_residuals(port_tree, portTransformed, targetColumn)
# math_tree = random_forests(mathTransformed, targetColumn, featureColumns, 42, 7, 3, 2)
# math_tree_resid = return_residuals(math_tree, mathTransformed, targetColumn)
# print('Done with tree')

# # Get KNN residuals
# knn_model = doKNN(portTransformed, targetColumn, featureColumns)
# port_knn_resid = return_residuals(knn_model, portTransformed, targetColumn)
# math_knn_resid = return_residuals(knn_model, mathTransformed, targetColumn)
# print('Done with KNN')


# portS = ['Portuguese' for i in range(4 * 65)]
# mathS = ['Math' for i in range(4 * 40)]

# model = []
# model += ['Linear Regression' for i in range(65)]
# model += ['SVM' for i in range(65)]
# model += ['Decision Tree' for i in range(65)]
# model += ['KNN' for i in range(65)]
# model += ['Linear Regression' for i in range(40)]
# model += ['SVM' for i in range(40)]
# model += ['Decision Tree' for i in range(40)]
# model += ['KNN' for i in range(40)]

# errors =  port_lm_resid + port_svm_resid + port_tree_resid + port_knn_resid
# errors += math_lm_resid + math_svm_resid + math_tree_resid + math_knn_resid
# subject = portS + mathS

# d_port = {'Model':model, 'Residual Error': errors, 'Subject': subject}
# df_port = pd.DataFrame(data=d_port)
# ax = sns.swarmplot(x = 'Model', y = 'Residual Error', hue='Subject', data=df_port, palette="Set2", split=True)
# ax.set_title('Residuals by Model')
# ax.grid(False)
# plt.show()

d = {'Feature' : feat, 'Importance' : imp}
df = pd.DataFrame(data=d)
ax = sns.barplot(x='Importance', y='Feature', data=df)
ax.set_title('Predicting Portuguese G1 Feature Importances')
ax.grid(False)
plt.show()

# Math:
# n_estimators: 46
# max_depth: 3
# min_samples_split: 4
# min_samples_leaf: 2
# r2:  0.12

# n_estimators: 42
# max_depth: 7
# min_samples_split: 3
# min_samples_leaf: 2
# MSE: -9.42

# n_estimators: 49
# max_depth: 5
# min_samples_split: 3
# min_samples_leaf: 1
# MAE: -2.53

# Portuguese:
# n_estimators: 42
# max_depth: 9
# min_samples_split: 4  
# min_samples_leaf: 1
# r2:  0.21

# n_estimators: 43
# max_depth: 9
# min_samples_split: 3
# min_samples_leaf: 2
# MSE: -5.26

# n_estimators: 43
# max_depth: 7
# min_samples_split: 2
# min_samples_leaf: 2
# MAE: -1.80




