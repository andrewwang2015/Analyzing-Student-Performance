import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
sns.set(style="whitegrid")
sns.set_color_codes("pastel")

mathTransformed = pd.read_csv('Datasets/math_transformed.csv')
portTransformed = pd.read_csv('Datasets/port_transformed.csv')

mathTransformed['diff'] = mathTransformed['G3'] - mathTransformed['G1']
portTransformed['diff'] = portTransformed['G3'] - portTransformed['G1']

targetColumn = 'diff'
featureColumns = [i for i in list(portTransformed) if i not in ['diff', 'G1', 'G2', 'G3']]

def random_forests(dataset, target_col, feat_cols):
    model = RandomForestRegressor()
    n_features = 7
    parameters = {'n_estimators' : [30],
                  'max_depth' : [5],
                  'min_samples_split' : [5],
                  'min_samples_leaf' : [7]}
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

#best_model, imp, feat = random_forests(mathTransformed, targetColumn, featureColumns)
best_model, imp, feat = random_forests(portTransformed, targetColumn, featureColumns)
# print('Math')
# random_forests(mathTransformed, targetColumn, featureColumns)
# print('Port')
# random_forests(portTransformed, targetColumn, featureColumns)

d = {'Feature' : feat, 'Importance' : imp}
df = pd.DataFrame(data=d)
ax = sns.barplot(x='Importance', y='Feature', data=df)
ax.set_title('Predicting Portuguese G1 to G3 Feature Importances')
ax.grid(False)
plt.show()

# math
# 0.0378459491016
# n_estimators: 24
# max_depth: 4
# min_samples_split: 5
# min_samples_leaf: 5

# port
# -0.0466857791582
# n_estimators: 30
# max_depth: 5
# min_samples_split: 5
# min_samples_leaf: 7







