"""
best params for this dataset
{'C': 0.1, 'loss': 'squared_hinge', 'penalty': 'l2'}
"""


import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV


param_dict = {'penalty': ['l1', 'l2'],
              'loss': ['hinge', 'squared_hinge'],
              'C': [0.0001, 0.001, 0.1, 1, 10, 100]}


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


svc_clf = LinearSVC()

search = GridSearchCV(estimator=svc_clf,
                      param_grid=param_dict,
                      scoring='accuracy',
                      n_jobs=-1,
                      refit=True,
                      return_train_score=True,
                      cv=10)


search.fit(X_train, y_train)

print(search.best_params_)

