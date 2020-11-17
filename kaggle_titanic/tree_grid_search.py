

import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


grid_params ={'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': range(1, 12),
              'min_samples_split': range(2, 25),
              'min_samples_leaf': range(1,12), 
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]}



titanic_df = pd.read_csv('dataset/train.csv')


def make_df(input_df:pd.DataFrame, columns:list) -> pd.DataFrame:
    return(input_df[columns])


def gen_to_float(x:str) -> float:
    if x == 'male':
        return(0)
    else:
        return(1)

    
  
# features prep
train_data = make_df(titanic_df, ['Pclass', 'Sex', 'Age'])
train_data['Sex'] = train_data['Sex'].map(lambda x: gen_to_float(x))
train_data = train_data.fillna(-1)

# target prep
mark = titanic_df['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(train_data, mark,
                                                    test_size=0.25, random_state=0)

# model learning

tree_clf = DecisionTreeClassifier()

search = GridSearchCV(estimator=tree_clf,
                      param_grid=grid_params,
                      scoring='accuracy',
                      n_jobs=-1,
                      refit=True,
                      return_train_score=True,
                      cv=10)


search.fit(X_train, y_train)

print(search.best_params_)