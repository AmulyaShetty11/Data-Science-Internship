import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/Admin/Desktop/titanic Amulya/titanic dataset.csv")
print(df)


#preparing X and Y
X = df.drop('Ticket', axis=1)
X = X.drop('PassengerId',axis=1)
X = X.drop('Cabin',axis=1)
X = X.drop('Name',axis=1)
X = X.drop('Survived',axis=1)
Y = df['Survived']
# print(X)
# print(Y)


#categorical to numerical
le = LabelEncoder()
le.fit(X['Sex'])
X['Sex'] = le.transform(df['Sex'])

le = LabelEncoder()
le.fit(X['Embarked'])
X['Embarked'] = le.transform(df['Embarked'])
# print(X)

#missing values
X['Pclass'].fillna((X['Pclass'].median()), inplace = True)
X['Sex'].fillna((X['Sex'].mean()), inplace = True)
X['Age'].fillna((X['Age'].max()), inplace = True)
X['SibSp'].fillna((X['SibSp'].min()), inplace = True)
X['Parch'].fillna((X['Parch'].min()), inplace = True)
X['Fare'].fillna((X['Fare'].min()), inplace = True)
X['Embarked'].fillna((X['Embarked'].min()), inplace = True)
# print(X)

from collections import Counter
# print(Counter(Y))

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X, Y= ros.fit_resample(X, Y)
# print(Counter(Y))


#feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X, Y)
# print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(7).plot(kind='barh')
# plt.show()


#identifying outliers by plotting
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['Fare'])
# plt.show()

#dealing with outliers using interquantile reange
# print(X['Fare'])
Q1 = X['Fare'].quantile(0.25)
Q2 = X['Fare'].quantile(0.75)

IQR = Q2 - Q1
# print(IQR)

upper = Q2 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['Fare']<lower].values
out2 = X[X['Fare']>upper].values

X['Fare'].replace(out1, lower, inplace=True)
X['Fare'].replace(out2, upper, inplace=True)
# print(X['Fare'])


#training
ran = RandomForestClassifier()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
ran.fit(x_train, y_train)
y_pred1 = ran.predict(x_test)
print(accuracy_score(y_test, y_pred1)*100)