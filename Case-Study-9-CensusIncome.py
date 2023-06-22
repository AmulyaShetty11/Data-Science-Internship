import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

df = pd.read_csv("C:/Users/admin/census income Amulya/adult.csv")
# print(df)

#Missing values
# print(df.isnull().sum())


# Replaing ? ---? NAN from dataset
df.replace("?", np.nan, inplace = True)
# print(df)

#missing values after filling with NAN
# print(df.isnull().sum())


#dropping unwanted attributes
X = df.drop('income', axis = 1)
X = X.drop('capital.gain', axis = 1)
Y = df['income']

# print(X.isnull().sum())

# Categorical to numerical convert
le = LabelEncoder()
le.fit(X['workclass'])
X['workclass'] = le.transform(X['workclass'])

le = LabelEncoder()
le.fit(X['education'])
X['education'] = le.transform(X['education'])

le = LabelEncoder()
le.fit(X['marital.status'])
X['marital.status'] = le.transform(X['marital.status'])

le = LabelEncoder()
le.fit(X['relationship'])
X['relationship'] = le.transform(X['relationship'])

le = LabelEncoder()
le.fit(X['occupation'])
X['occupation'] = le.transform(X['occupation'])

le = LabelEncoder()
le.fit(X['race'])
X['race'] = le.transform(X['race'])

le = LabelEncoder()
le.fit(X['sex'])
X['sex'] = le.transform(X['sex'])

le = LabelEncoder()
le.fit(X['native.country'])
X['native.country'] = le.transform(X['native.country'])

# print(X)

# print(X.isnull().sum())

# Filling Missing values
X['workclass'].fillna((X['workclass'].mean()), inplace=True)
X['occupation'].fillna((X['occupation'].mean()), inplace=True)
X['native.country'].fillna((X['native.country'].mean()), inplace=True)

# print(X.isnull().sum())


# Feature Selection 1
# bestfeatures = SelectKBest(score_func=chi2, k = 'all')
# fit = bestfeatures.fit(X,Y)  # training
# dfscores = pd.DataFrame(fit.scores_) # scores store
# dfcolumns = pd.DataFrame(X.columns)  # 4 col value scores
# featurescores = pd.concat([dfcolumns, dfscores], axis=1) # concat scores and cols
# featurescores.columns = ['attributes', 'Score'] #label this cols for results
# print(featurescores)


# Feature Selection 2
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print(model.feature_importances_)
# feat_importance = pd.Series(model.feature_importances_, index=X.columns)
# feat_importance.nlargest(13).plot(kind='barh')
# plt.show()

#checking the size of every class
# print(Counter(Y))

# print("\n")

#Balancing the dataset
from imblearn.over_sampling import RandomOverSampler  # -----> OverSampling 2
sms = RandomOverSampler(random_state=0)
X,Y = sms.fit_resample(X, Y)
# print(Counter(Y))


#Identifying outliers by ploting GRAPH
# sns.boxplot(X['age'])
# plt.show()

# sns.boxplot(df['fnlwgt'])
# plt.show()

# sns.boxplot(df['education.num'])  #--->easy
# plt.show()

# sns.boxplot(df['capital.loss'])  #--->not neccessary
# plt.show()

# sns.boxplot(df['hours.per.week'])
# plt.show()


# print(X['age'])
Q1 = X['age'].quantile(0.25)
Q3 = X['age'].quantile(0.75)

IQR = Q3 - Q1
# print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['age'] < lower].values
out2 = X[X['age'] > upper].values

X['age'].replace(out1, lower, inplace=True)
X['age'].replace(out2, upper, inplace=True)

# print(X['age'])

sns.boxplot(X['age'])
# plt.show()


# print(X['fnlwgt'])
Q1 = X['fnlwgt'].quantile(0.25)
Q3 = X['fnlwgt'].quantile(0.75)

IQR = Q3 - Q1
# print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['fnlwgt'] < lower].values
out2 = X[X['fnlwgt'] > upper].values

X['fnlwgt'].replace(out1, lower, inplace=True)
X['fnlwgt'].replace(out2, upper, inplace=True)

# print(X['fnlwgt'])

sns.boxplot(X['fnlwgt'])
# plt.show()


# print(X['education.num'])
Q1 = X['education.num'].quantile(0.25)
Q3 = X['education.num'].quantile(0.75)

IQR = Q3 - Q1
# print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['education.num'] < lower].values
out2 = X[X['education.num'] > upper].values

X['education.num'].replace(out1, lower, inplace=True)
X['education.num'].replace(out2, upper, inplace=True)

# print(X['education.num'])

sns.boxplot(X['education.num'])
# plt.show()


# print(X['hours.per.week'])
Q1 = X['hours.per.week'].quantile(0.25)
Q3 = X['hours.per.week'].quantile(0.75)

IQR = Q3 - Q1
# print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['hours.per.week'] < lower].values
out2 = X[X['hours.per.week'] > upper].values

X['hours.per.week'].replace(out1, lower, inplace=True)
X['hours.per.week'].replace(out2, upper, inplace=True)

# print(X['hours.per.week'])

sns.boxplot(X['hours.per.week'])
# plt.show()


# print(X['capital.loss'])
Q1 = X['capital.loss'].quantile(0.25)
Q3 = X['capital.loss'].quantile(0.75)

IQR = Q3 - Q1
# print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

# print(upper)
# print(lower)

out1 = X[X['capital.loss'] < lower].values
out2 = X[X['capital.loss'] > upper].values

X['capital.loss'].replace(out1, lower, inplace=True)
X['capital.loss'].replace(out2, upper, inplace=True)

# print(X['capital.loss'])

sns.boxplot(X['capital.loss'])
# plt.show()


#TRAINING MODEL
ran = RandomForestClassifier()

X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,random_state=0,test_size=0.3)
ran.fit(X_train,Y_train)

Y_pred1 = ran.predict(X_test)
print("Predication is: ")
print(accuracy_score(Y_test,Y_pred1)*100)