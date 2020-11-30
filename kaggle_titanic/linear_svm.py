'''
A good improvement for SVC model - 0.76555
'''

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


svc_clf = LinearSVC(C=0.1, loss='squared_hinge', penalty='l2')

cv = KFold(n_splits=10, shuffle=True, random_state=7)

scores = cross_val_score(svc_clf, X_train, y_train,
                         scoring='accuracy', cv=cv,
                         n_jobs=-1)


svc_clf.fit(X_train, y_train)

submit_data = pd.read_csv('dataset/test.csv')

submit_df = make_df(submit_data, ['Pclass', 'Sex', 'Age'])
submit_df['Sex'] = submit_df['Sex'].map(lambda x: gen_to_float(x))
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(submit_df)
X = imputer.transform(submit_df)

pred = svc_clf.predict(X)

pred.astype(int)


sub_df = pd.DataFrame()

sub_df['PassengerId'] = submit_data['PassengerId']
sub_df['Survived'] = pd.Series(pred)

sub_df.to_csv('submit_svc.csv', index=False)