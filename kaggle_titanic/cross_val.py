"""
A basic cross validation to use later in the project
"""


import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold


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

cv = KFold(n_splits=10, shuffle=True, random_state=7)

scores = cross_val_score(tree_clf, X_train, y_train,
                         scoring='f1', cv=cv,
                         n_jobs=-1)

print(scores)
print(np.mean(scores))