import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("C:/Users/Admin/Desktop/black friday Amulya/black_friday.csv")
# print(df)

#preparing X and Y
X = df.drop('User_ID', axis=1)
X = X.drop('Age',axis=1)
X = X.drop('Occupation',axis=1)
X = X.drop('Stay_In_Current_City_Years',axis=1)
X = X.drop('Purchase',axis=1)
Y = df['Purchase']
# print(X)
# print(Y)


#categorical to numerical
le = LabelEncoder()
le.fit(X['Product_ID'])
X['Product_ID'] = le.transform(df['Product_ID'])

le = LabelEncoder()
le.fit(X['Gender'])
X['Gender'] = le.transform(df['Gender'])

le = LabelEncoder()
le.fit(X['City_Category'])
X['City_Category'] = le.transform(df['City_Category'])


#missing values
X['Product_ID'].fillna((X['Product_ID'].median()), inplace = True)
X['Gender'].fillna((X['Gender'].mean()), inplace = True)
X['City_Category'].fillna((X['City_Category'].max()), inplace = True)
X['Marital_Status'].fillna((X['Marital_Status'].min()), inplace = True)
X['Product_Category_1'].fillna((X['Product_Category_1'].min()), inplace = True)
X['Product_Category_2'].fillna((X['Product_Category_2'].min()), inplace = True)
X['Product_Category_3'].fillna((X['Product_Category_3'].min()), inplace = True)
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
# from sklearn.ensemble import ExtraTreesRegressor
# import matplotlib.pyplot as plt
#
# model = ExtraTreesRegressor()
# model.fit( X, Y)
# # model.feature_importances_
# feat_imps = pd.Series(model.feature_importances_, index=X.columns)
# feat_imps.nlargest(7).plot(kind = 'barh')
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