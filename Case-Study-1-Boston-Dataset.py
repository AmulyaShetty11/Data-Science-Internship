import pandas  as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/admin/Desktop/Boston Amulya/HousingData.csv")
# print(df.isnull().sum())

# preapering X and Y
X = df.drop('ZN', axis=1)
X = X.drop('CHAS', axis=1)
X = X.drop('MEDV', axis=1)
Y = df['MEDV']

# print(X)
# print(Y)

#Missing values
X['CRIM'].fillna((X['CRIM'].max()), inplace=True)
X['INDUS'].fillna((X['INDUS'].max()), inplace=True)
X['AGE'].fillna((X['AGE'].max()), inplace=True)
X['LSTAT'].fillna((X['LSTAT'].max()), inplace=True)
#
# print(X.isnull().sum())

# Feature Selection 1
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression
# bestfeature = SelectKBest(score_func=f_regression,  k='all')
# fit = bestfeature.fit(X, Y)
# dfscorces = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featurescores = pd.concat((dfcolumns, dfscorces), axis=1)
# print(featurescores)

# Feature Selection 2
# model = ExtraTreesRegressor()
# model.fit(X,Y)
# model.feature_importances_
# feat_imps = pd.Series(model.feature_importances_, index=X.columns)
# feat_imps.nlargest(11).plot(kind = 'barh')
# plt.show()


#training the model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

gbm = GradientBoostingRegressor()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
gbm.fit(x_train, y_train)

y_pred = gbm.predict(x_test)
gbm.fit(x_train,y_train )
print(r2_score(y_test, y_pred))
