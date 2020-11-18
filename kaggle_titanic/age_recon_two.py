"""
Replacing all missing values with median

Gives result 0.59090 that is an absolute bottom position on the Leaderboard
"""

import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer 


# создам объект класса SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')


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
   
# features prep
train_data = make_df(titanic_df, ['Pclass', 'Sex', 'Age'])
train_data['Sex'] = train_data['Sex'].map(lambda x: gen_to_float(x))


imputer.fit(train_data)
train_data = imputer.transform(train_data)

mark = titanic_df['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(train_data, mark,
                                                    test_size=0.25, random_state=0)


# model learning
tree_clf = DecisionTreeClassifier(criterion='gini',
                                  max_depth=8,
                                  max_features='auto',
                                  max_leaf_nodes=7,
                                  min_samples_leaf=5,
                                  min_samples_split=24,
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

imputer.fit(submit_df)
X = imputer.transform(submit_df)

pred = tree_clf.predict(X)

pred.astype(int)


sub_df = pd.DataFrame()

sub_df['PassengerId'] = submit_data['PassengerId']
sub_df['Survived'] = pd.Series(pred)

sub_df.to_csv('submit2.csv', index=False)
 
 
