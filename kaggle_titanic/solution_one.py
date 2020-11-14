"""
Just add two more features from the dataset to my baseline solution:
- 'SibSp';
- 'Parch'
Accuracy: 0.72966
Position: ~15750
"""


import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


titanic_df = pd.read_csv('dataset/train.csv')


def make_df(input_df:pd.DataFrame, columns:list) -> pd.DataFrame:
    return(input_df[columns])


def gen_to_float(x:str) -> float:
    if x == 'male':
        return(0)
    else:
        return(1)

    
  
# features prep
train_data = make_df(titanic_df, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
train_data['Sex'] = train_data['Sex'].map(lambda x: gen_to_float(x))
train_data = train_data.fillna(-1)

# target prep
mark = titanic_df['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(train_data, mark,
                                                    test_size=0.25, random_state=0)

# model learning

tree_clf = DecisionTreeClassifier()

tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)


# submit

submit_data = pd.read_csv('dataset/test.csv')

submit_df = make_df(submit_data, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
submit_df['Sex'] = submit_df['Sex'].map(lambda x: gen_to_float(x))
submit_df = submit_df.fillna(-1)

pred = tree_clf.predict(submit_df)

pred.astype(int)


sub_df = pd.DataFrame()

sub_df['PassengerId'] = submit_data['PassengerId']
sub_df['Survived'] = pd.Series(pred)

sub_df.to_csv('submit.csv', index=False)