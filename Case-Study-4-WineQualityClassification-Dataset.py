import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("C:/Users/admin/Desktop/winequalityN Amulya/winequalityN.csv")

le = LabelEncoder()
le.fit(df['type'])
df['type'] = le.transform(df['type'])

df['fixed acidity'].fillna(df['fixed acidity'].mean(), inplace=True)
df['volatile acidity'].fillna(df['volatile acidity'].median(), inplace=True)
df['citric acid'].fillna(df['citric acid'].max(), inplace=True)
df['residual sugar'].fillna(df['residual sugar'].min(), inplace=True)
df['chlorides'].fillna(df['chlorides'].min(), inplace=True)
df['free sulfur dioxide'].fillna(df['free sulfur dioxide'].min(), inplace=True)
df['total sulfur dioxide'].fillna(df['total sulfur dioxide'].min(), inplace=True)
df['density'].fillna(df['density'].min(), inplace=True)
df['pH'].fillna(df['pH'].min(), inplace=True)
df['sulphates'].fillna(df['sulphates'].min(), inplace=True)
df['alcohol'].fillna(df['alcohol'].min(), inplace=True)
df['quality'].fillna(df['quality'].min(), inplace=True)

X = df.drop('quality', axis=1)
Y = df['quality']

from collections import Counter

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X, Y= ros.fit_resample(X, Y)

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print(model.feature_importances_)
# feat_importance = pd.Series(model.feature_importances_, index=X.columns)
# feat_importance.nlargest(12).plot(kind='barh')
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['fixed acidity'])
# plt.show()
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['volatile acidity'])
# plt.show()
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['citric acid'])
# plt.show()
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['residual sugar'])
# plt.show()
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['chlorides'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['free sulfur dioxide'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['total sulfur dioxide'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['pH'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['alcohol'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['density'])
# plt.show()

print(X['density'])
Q1 = X['density'].quantile(0.25)
Q3 = X['density'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 = X[X['density'] < lower].values
out2 = X[X['density'] > upper].values
X['density'].replace(out1,lower,inplace=True)
X['density'].replace(out2,upper,inplace=True)
print(X['density'])
# sns.boxplot(X['density'])
# plt.show()


print(X['pH'])
Q1 = X['pH'].quantile(0.25)
Q3 = X['pH'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 = X[X['pH'] < lower].values
out2 = X[X['pH'] > upper].values
X['pH'].replace(out1,lower,inplace=True)
X['pH'].replace(out2,upper,inplace=True)
print(X['pH'])

# sns.boxplot(X['pH'])
# plt.show()

print(X['total sulfur dioxide'])
Q1 = X['total sulfur dioxide'].quantile(0.25)
Q3 = X['total sulfur dioxide'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 = X[X['total sulfur dioxide'] < lower].values
out2 = X[X['total sulfur dioxide'] > upper].values
X['total sulfur dioxide'].replace(out1,lower,inplace=True)
X['total sulfur dioxide'].replace(out2,upper,inplace=True)
print(X['total sulfur dioxide'])
# sns.boxplot(X['total sulfur dioxide'])
# plt.show()

print(X['free sulfur dioxide'])
Q1 = X['free sulfur dioxide'].quantile(0.25)
Q3 = X['free sulfur dioxide'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)
out1 = X[X['free sulfur dioxide'] < lower].values
out2 = X[X['free sulfur dioxide'] > upper].values
X['free sulfur dioxide'].replace(out1,lower,inplace=True)
X['free sulfur dioxide'].replace(out2,upper,inplace=True)
print(X['free sulfur dioxide'])

# sns.boxplot(X['free sulfur dioxide'])
# plt.show()

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print(model.feature_importances_)
# feat_importance = pd.Series(model.feature_importances_, index=X.columns)
# feat_importance.nlargest(7).plot(kind='barh')
# plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ran = RandomForestClassifier()


X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,random_state=0,test_size=0.3)
ran.fit(X_train,Y_train)

Y_pred1 = ran.predict(X_test)
print(accuracy_score(Y_test,Y_pred1)*100)
