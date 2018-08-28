import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
sns.set(style="whitegrid")
sns.set_color_codes("pastel")

# Read our dataset
mathTransformed = pd.read_csv('Datasets/math_transformed.csv')
portTransformed = pd.read_csv('Datasets/port_transformed.csv')

# We are predicting quarter 1 grades
targetColumn = 'G1'
# We don't want to use G1, G2, or G3 to predict G1
featureColumns = [i for i in list(portTransformed) if i not in ['G1', 'G2', 'G3']]

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

def doKNN(dataset, target_col, feat_cols):
    model = KNeighborsRegressor()
    parameters = {'n_neighbors' : np.arange(10, 50, 2),
                  'weights'     : ['uniform', 'distance'],
                  'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute']}
    test_scores = []
    X = dataset[feat_cols]
    y = dataset[target_col]
    for score in ['r2','neg_mean_squared_error', 'neg_mean_absolute_error']:
        clf = GridSearchCV(model, parameters, cv=5, scoring=score)
        clf.fit(X,y)
        test_scores.append(clf.best_score_)
        print('Using {} neighbors'.format(clf.best_estimator_.n_neighbors))
    return clf.best_estimator_, test_scores

print('Math:')
best_model, test_scores = doKNN(mathTransformed, targetColumn, featureColumns)
print("KNN  r2: %0.2f" % (test_scores[0]))
print("KNN MSE: %0.2f" % (test_scores[1]))
print("KNN MAE: %0.2f" % (test_scores[2]))
math_resid = return_residuals(best_model, mathTransformed, targetColumn)

print('Portugeuese:')
best_model, test_scores = doKNN(portTransformed, targetColumn, featureColumns)
print("KNN  r2: %0.2f" % (test_scores[0]))
print("KNN MSE: %0.2f" % (test_scores[1]))
print("KNN MAE: %0.2f" % (test_scores[2]))
port_resid = return_residuals(best_model, portTransformed, targetColumn)

errors = port_resid + math_resid
subject = ['Portuguese' for i in range(65)] + ['Math' for i in range(40)]

d_port = {'Residual Error': errors, 'Subject': subject}
df_port = pd.DataFrame(data=d_port)
ax = sns.swarmplot(x='Subject', y='Residual Error', data=df_port, palette='Set2')
ax.set_title('KNN Residuals')
ax.grid(False)
plt.show()



