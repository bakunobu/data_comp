"""
Best params for SVC
 best params: {'lernel': 'poly', 'C': 0.001, 'degree': 2, 'gamma': 0.1}

score ~0.62200
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

titanic_df = pd.read_csv('dataset/train.csv')



def make_df(input_df:pd.DataFrame, columns:list) -> pd.DataFrame:
    return(input_df[columns])


def gen_to_float(x:str) -> int:
    if x == 'male':
        return(0)
    else:
        return(1)
 
    
def embarked_to_num(x:str) -> int:
    if x == 'C':
        return(0)
    elif x == 'Q':
        return(1)
    elif x == 'S':
        return(2)
    else:
        return(np.NaN)


train_data = make_df(titanic_df, ['Pclass', 'Sex', 'Age', 'Embarked',
                                  'Name', 'SibSp', 'Parch', 'Survived'])


train_data = train_data.dropna()


target = train_data['Survived']
train_data.drop(columns=['Survived'], inplace=True)


titles = set()


for name in train_data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())


title_dict = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Dona': 'Royalty',
    'Sir': 'Royalty',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'the Countess': 'Royalty',
    'Mme': 'Mrs',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Lady': 'Royalty'
}


def add_title(x: str) -> str:
    name = x.split(',')[1].split('.')[0].strip()
    return(title_dict[name])


train_data['Title'] = train_data['Name'].map(lambda x: add_title(x))
train_data.drop(columns=['Name'], inplace=True)
train_data['Embarked'] = train_data['Embarked'].map(lambda x: embarked_to_num(x))

train_data['Sex'] = train_data['Sex'].map(lambda x: gen_to_float(x))


title_dummies = pd.get_dummies(train_data['Title'], prefix='Title')
train_data = pd.concat([train_data, title_dummies], axis=1)
train_data.drop(columns=['Title'], inplace=True)



X_train, X_test, y_train, y_test = train_test_split(train_data, target,
                                                    test_size=0.25, random_state=0)


poly_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_svc', SVC(kernel='poly', C=0.001,
                     degree=2, gamma=0.1))
])

poly_svm_clf.fit(X_train, y_train)



submit_df = pd.read_csv('dataset/test.csv')


submit_data = make_df(submit_df, ['Pclass', 'Sex', 'Age', 'Embarked',
                                  'Name', 'SibSp', 'Parch',])

submit_data['Title'] = submit_data['Name'].map(lambda x: add_title(x))
submit_data.drop(columns=['Name'], inplace=True)
submit_data['Embarked'] = submit_data['Embarked'].map(lambda x: embarked_to_num(x))

submit_data['Sex'] = submit_data['Sex'].map(lambda x: gen_to_float(x))


title_dummies = pd.get_dummies(submit_data['Title'], prefix='Title')
submit_data = pd.concat([submit_data, title_dummies], axis=1)
submit_data.drop(columns=['Title'], inplace=True)



submit_data = submit_data.fillna(-1)

pred = poly_svm_clf.predict(submit_data)

pred.astype(int)


sub_df = pd.DataFrame()

sub_df['PassengerId'] = submit_df['PassengerId']
sub_df['Survived'] = pd.Series(pred)

sub_df.to_csv('submit.csv', index=False)
