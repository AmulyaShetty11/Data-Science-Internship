import pandas as pd
from sklearn.datasets import load_iris

irs = load_iris()
# print(irs)
# print(irs.keys())
# print(irs.values())
# print(irs.data)
# print(irs.target)
# print(irs.feature_names)
# print(irs.target_names)
# print(irs.DESCR)

df=pd.read_csv("C:/Users/Admin/Desktop/iris Amulya/IRIS.csv")
print(df)
# print(df.head(10))
# print(df.tails())
# print(df.columns.values)
# print(df.describe())


#preparing X and Y
x= df.drop('species', axis=1)
y= df['species']
# print(x)
# print(y)


#feature selection (1)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest (score_func=chi2, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame (fit.scores_)
dfcolumns = pd.DataFrame (x.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['specs', 'score']
# print(featuresScores)

#feature selection (2)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index=x.columns)
feat_importance.nlargest(4).plot(kind='barh')
# plt.show()

#numerical to categorical
df['sepal_length']=pd.cut(df['sepal_length'],3,labels=['0','1','2'])
df['sepal_width']=pd.cut(df['sepal_width'],3,labels=['0','1','2'])
df['petal_length']=pd.cut(df['petal_length'],3,labels=['0','1','2'])
df['petal_width']=pd.cut(df['petal_width'],3,labels=['0','1','2'])
# print(df)


#missing values
# print(df.isnull().sum())

df['sepal_length'].fillna((df['sepal_length'].max()), inplace = True)
df['sepal_width'].fillna((df['sepal_width'].max()), inplace = True)
df['petal_length'].fillna((df['petal_length'].max()), inplace = True)
df['petal_width'].fillna((df['petal_width'].min()), inplace = True)
# print(df.isnull().sum())
# print(df.count())

# imbalance in dataset
# over sampling and under sampling

a=(df['species']=='Iris-setosa').sum()
b=(df['species']=='Iris-versicolor').sum()
c=(df['species']=='Iris-virginica').sum()
# print('Iris-setosa=',a)
# print('Iris-versicolor=',b)
# print('Iris-virginica=',c)

from collections import Counter
# print(Counter(y))

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x, y)
# print(Counter(y))

# from imblearn.over_sampling import SMOTE
# sms=SMOTE(random_state=0)
# x, y = sms.fit_resample(x, y)

# from imblearn.under_sampling import RandomUnderSampler
# rus=RandomUnderSampler(random_state=0)
# x, y = rus.fit_resample(x, y)


# #identifying outliers by plotting
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(df['sepal_length'])
# # plt.show()
#
# #dealing with outliers using interquantile reange
# # print(df['sepal_length'])
# Q1 = df['sepal_length'].quantile(0.25)
# Q2 = df['sepal_length'].quantile(0.75)
#
# IQR = Q2 - Q1
# # print(IQR)
#
# upper = Q2 + 1.5*IQR
# lower = Q1 - 1.5*IQR
#
# # print(upper)
# # print(lower)
#
# out1 = df[df['sepal_length']<lower].values
# out2 = df[df['sepal_length']>upper].values
#
# df['sepal_length'].replace(out1, lower, inplace=True)
# df['sepal_length'].replace(out2, upper, inplace=True)
# # print(df['sepal_length'])


#training
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression()
pca = PCA(n_components=2)

x = df.drop('species', axis=1)
y = df['species']

pca.fit(x)
x = pca.transform(x)

print(x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, test_size=0.3)
logr.fit(X_train, Y_train)

Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred)*100)