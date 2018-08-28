import pandas as pd
import numpy as np
import statistics as stat
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

mathTransformed['diff'] = mathTransformed['G3'] - mathTransformed['G1']
portTransformed['diff'] = portTransformed['G3'] - portTransformed['G1']

targetColumn = 'diff'
featureColumns = [i for i in list(portTransformed) if i not in ['diff', 'G1', 'G2', 'G3']]

def return_residuals(model, df, col_to_predict):
    '''
    Using a 90/10 split of data, returns list of residuals for predictions
    of testing set.
    '''
    all_columns = list(df)
    all_X = [i for i in all_columns if i not in ['diff', 'G1', 'G2', 'G3']]
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

def random_forests(dataset, target_col, feat_cols, NE, MD, MSS, MSL):
    model = RandomForestRegressor(NE, max_depth=MD, min_samples_split=MSS, min_samples_leaf=MSL)
    return model

# Get tree residuals
port_tree = random_forests(portTransformed, targetColumn, featureColumns, 24, 4, 5, 5)
port_tree_resid = return_residuals(port_tree, portTransformed, targetColumn)
math_tree = random_forests(mathTransformed, targetColumn, featureColumns, 42, 7, 3, 2)
math_tree_resid = return_residuals(math_tree, mathTransformed, targetColumn)

actualDiffMath = list(mathTransformed['diff'])
actualDiffPort = list(portTransformed['diff'])

median_math = stat.median(actualDiffMath)
median_port = stat.median(actualDiffPort)
mean_math = sum(actualDiffMath) / len(actualDiffMath)
mean_port = sum(actualDiffPort)/ len(actualDiffPort)
math_median_resid = [k - median_math for k in np.random.choice(actualDiffMath, 40)]
port_median_resid = [k - median_port for k in np.random.choice(actualDiffPort, 65)]

math_mean_resid = [k - mean_math for k in np.random.choice(actualDiffMath, 40)]
port_mean_resid = [k - mean_port for k in np.random.choice(actualDiffPort, 65)]

portS = ['Portuguese' for i in range(3 * 65)]
mathS = ['Math' for i in range(3 * 40)]

model = []
model += ['Regression Tree' for i in range(65)]
model += ['Median' for i in range(65)]
model += ['Mean' for i in range(65)]
model += ['Regression Tree' for i in range(40)]
model += ['Median' for i in range(40)]
model += ['Mean' for i in range(40)]

errors =  port_tree_resid + port_median_resid + port_mean_resid + math_tree_resid + math_median_resid + math_mean_resid
subject = portS + mathS

print(len(model), len(errors), len(subject))

d_port = {'Model':model, 'Residual Error': errors, 'Subject': subject}
df_port = pd.DataFrame(data=d_port)
ax = sns.swarmplot(x = 'Model', y = 'Residual Error', hue='Subject', data=df_port, palette="Set2", split=True)
ax.set_title('Residuals by Model')
ax.grid(False)
plt.show()






