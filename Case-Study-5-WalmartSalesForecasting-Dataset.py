import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv("C:/Users/admin/Desktop/Data Science Amulya/train.csv")
# print(df)

#preapering X and Y
X = df.drop('Weekly_Sales', axis=1)
X = X.drop('Date', axis=1)
Y = df['Weekly_Sales']

# print(X)
# print(Y)

# replacing values
le = LabelEncoder()
le.fit(X['IsHoliday'])
X['IsHoliday'] = le.transform(X['IsHoliday'])
# print(X)

#Missing values
# print(X.isnull().sum())

# Feature Selection 1
# bestfeature = SelectKBest(score_func=f_regression,  k='all')
# fit = bestfeature.fit(X, Y)
# dfscorces = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featurescores = pd.concat((dfcolumns, dfscorces), axis=1)
# print(featurescores)



# Feature Selection 2
# model = ExtraTreesRegressor()
# model.fit(X,Y)
# print(model.feature_importances_)
#
# feat_importace =pd.Series(model.feature_importances_, index=X.columns)
# feat_importace.nlargest(4).plot(kind='barh')
# plt.show()

#training model

gbm = GradientBoostingRegressor()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
gbm.fit(x_train, y_train)

y_pred = gbm.predict(x_test)
gbm.fit(x_train,y_train )
print(r2_score(y_test, y_pred))
