"""
Replacing all the records, that contains NaN in from the set

Gives result 0.77272 that is ~ 10380 position on the Leaderboard

New best_params (not really) - gives result 0.73444
{'max_depth': 6, 'max_leaf_nodes': 7, 'min_samples_leaf': 3, 'min_samples_split': 22}
"""

import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV



# берем датасет
titanic_df = pd.read_csv('dataset/train.csv')


def make_df(input_df:pd.DataFrame, columns:list) -> pd.DataFrame:
    return(input_df[columns])


def gen_to_float(x:str) -> float:
    if x == 'male':
        return(0)
    else:
        return(1)


# дропаем значения
titanic_nona = titanic_df.dropna(subset=['Age'])
   
# features prep
train_data = make_df(titanic_nona, ['Pclass', 'Sex', 'Age'])
train_data['Sex'] = train_data['Sex'].map(lambda x: gen_to_float(x))

mark = titanic_nona['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(train_data, mark,
                                                    test_size=0.25, random_state=0)


"""
grid_param = {'max_depth': [6, 7, 8, 9, 10],
              'min_samples_split': [22, 23, 24, 25, 26, 27],
              'min_samples_leaf': [3, 4, 5, 6, 7],
              'max_leaf_nodes': [5, 6, 7, 8, 9]}


tree_clf = DecisionTreeClassifier()


search = GridSearchCV(estimator=tree_clf,
                      param_grid=grid_param,
                      scoring='accuracy',
                      n_jobs=-1,
                      refit=True,
                      return_train_score=True,
                      cv=10)


search.fit(X_train, y_train)

print(search.best_params_)


"""
# model learning
tree_clf = DecisionTreeClassifier(criterion='gini',
                                  max_depth=6,
                                  max_features='auto',
                                  max_leaf_nodes=7,
                                  min_samples_leaf=3,
                                  min_samples_split=22,
                                  splitter='best')

cv = KFold(n_splits=10, shuffle=True, random_state=7)

scores = cross_val_score(tree_clf, X_train, y_train,
                         scoring='accuracy', cv=cv,
                         n_jobs=-1)

tree_clf.fit(X_train, y_train)

 
# submit

submit_data = pd.read_csv('dataset/test.csv')

submit_df = make_df(submit_data, ['Pclass', 'Sex', 'Age'])
submit_df['Sex'] = submit_df['Sex'].map(lambda x: gen_to_float(x))
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(submit_df)
X = imputer.transform(submit_df)

pred = tree_clf.predict(X)

pred.astype(int)


sub_df = pd.DataFrame()

sub_df['PassengerId'] = submit_data['PassengerId']
sub_df['Survived'] = pd.Series(pred)

sub_df.to_csv('submit.csv', index=False)

 
