import pandas as pd

df = pd.read_csv("C:/Users/admin/Desktop/UCI HAR/tidyData.csv")
# print(df)

X = df.drop('Subject', axis=1)
X = X.drop('Activity', axis=1)
Y = df['Activity']

# print(X)
# print(Y)

from collections import Counter
# print(Counter(Y))

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
# print(Counter(Y))


from matplotlib import pyplot as plt
import seaborn as sns

# sns.boxplot(X['Time domain:Bodyacceleration,standard deviration valueon Z axis'])
# plt.show()

# print(X['Time domain:Bodyacceleration,standard deviration valueon Z axis'])
Q1 = X['Time domain:Bodyacceleration,standard deviration valueon Z axis'].quantile(0.25)
Q3 = X['Time domain:Bodyacceleration,standard deviration valueon Z axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Bodyacceleration,standard deviration valueon Z axis'] < lower].values
out2 = X[X['Time domain:Bodyacceleration,standard deviration valueon Z axis'] > upper].values
X['Time domain:Bodyacceleration,standard deviration valueon Z axis'].replace(out1,lower,inplace=True)
X['Time domain:Bodyacceleration,standard deviration valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration,standard deviration valueon Z axis'])

# sns.boxplot(X['Time domain:Bodyacceleration,standard deviration valueon Z axis'])
# plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon X axis'])
# plt.show()

# print(X['Time domain:Gravityacceleration,mean valueon X axis'])
Q1 = X['Time domain:Gravityacceleration,mean valueon X axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,mean valueon X axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,mean valueon X axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,mean valueon X axis'] > upper].values
X['Time domain:Gravityacceleration,mean valueon X axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,mean valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,mean valueon X axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon X axis'])
plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon Y axis'])
# plt.show()

# print(X['Time domain:Gravityacceleration,mean valueon Y axis'])
Q1 = X['Time domain:Gravityacceleration,mean valueon Y axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,mean valueon Y axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,mean valueon Y axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,mean valueon Y axis'] > upper].values
X['Time domain:Gravityacceleration,mean valueon Y axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,mean valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,mean valueon Y axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon Y axis'])
# plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon Z axis'])
# plt.show()

# print(X['Time domain:Gravityacceleration,mean valueon Z axis'])
Q1 = X['Time domain:Gravityacceleration,mean valueon Z axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,mean valueon Z axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,mean valueon Z axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,mean valueon Z axis'] > upper].values
X['Time domain:Gravityacceleration,mean valueon Z axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,mean valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,mean valueon Z axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,mean valueon Z axis'])
# plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon X axis'])
# plt.show()
#
# print(X['Time domain:Gravityacceleration,standard deviration valueon X axis'])
Q1 = X['Time domain:Gravityacceleration,standard deviration valueon X axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,standard deviration valueon X axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,standard deviration valueon X axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,standard deviration valueon X axis'] > upper].values
X['Time domain:Gravityacceleration,standard deviration valueon X axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,standard deviration valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,standard deviration valueon X axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon X axis'])
# plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon Y axis'])
# plt.show()

# print(X['Time domain:Gravityacceleration,standard deviration valueon Y axis'])
Q1 = X['Time domain:Gravityacceleration,standard deviration valueon Y axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,standard deviration valueon Y axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,standard deviration valueon Y axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,standard deviration valueon Y axis'] > upper].values
X['Time domain:Gravityacceleration,standard deviration valueon Y axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,standard deviration valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,standard deviration valueon Y axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon Y axis'])
# plt.show()

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon Z axis'])
# plt.show()

# print(X['Time domain:Gravityacceleration,standard deviration valueon Z axis'])
Q1 = X['Time domain:Gravityacceleration,standard deviration valueon Z axis'].quantile(0.25)
Q3 = X['Time domain:Gravityacceleration,standard deviration valueon Z axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Gravityacceleration,standard deviration valueon Z axis'] < lower].values
out2 = X[X['Time domain:Gravityacceleration,standard deviration valueon Z axis'] > upper].values
X['Time domain:Gravityacceleration,standard deviration valueon Z axis'].replace(out1,lower,inplace=True)
X['Time domain:Gravityacceleration,standard deviration valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Gravityacceleration,standard deviration valueon Z axis'])

# sns.boxplot(X['Time domain:Gravityacceleration,standard deviration valueon Z axis'])
# plt.show()

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon X axis'])
# plt.show()

# print(X['Time domain:Bodyacceleration jerk,mean valueon X axis'])
Q1 = X['Time domain:Bodyacceleration jerk,mean valueon X axis'].quantile(0.25)
Q3 = X['Time domain:Bodyacceleration jerk,mean valueon X axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Bodyacceleration jerk,mean valueon X axis'] < lower].values
out2 = X[X['Time domain:Bodyacceleration jerk,mean valueon X axis'] > upper].values
X['Time domain:Bodyacceleration jerk,mean valueon X axis'].replace(out1,lower,inplace=True)
X['Time domain:Bodyacceleration jerk,mean valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,mean valueon X axis'])

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon X axis'])
# plt.show()

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon Y axis'])
# plt.show()

# print(X['Time domain:Bodyacceleration jerk,mean valueon Y axis'])
Q1 = X['Time domain:Bodyacceleration jerk,mean valueon Y axis'].quantile(0.25)
Q3 = X['Time domain:Bodyacceleration jerk,mean valueon Y axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Bodyacceleration jerk,mean valueon Y axis'] < lower].values
out2 = X[X['Time domain:Bodyacceleration jerk,mean valueon Y axis'] > upper].values
X['Time domain:Bodyacceleration jerk,mean valueon Y axis'].replace(out1,lower,inplace=True)
X['Time domain:Bodyacceleration jerk,mean valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,mean valueon Y axis'])

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon Y axis'])
# plt.show()

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon Z axis'])
# plt.show()

# print(X['Time domain:Bodyacceleration jerk,mean valueon Z axis'])
Q1 = X['Time domain:Bodyacceleration jerk,mean valueon Z axis'].quantile(0.25)
Q3 = X['Time domain:Bodyacceleration jerk,mean valueon Z axis'].quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
out1 = X[X['Time domain:Bodyacceleration jerk,mean valueon Z axis'] < lower].values
out2 = X[X['Time domain:Bodyacceleration jerk,mean valueon Z axis'] > upper].values
X['Time domain:Bodyacceleration jerk,mean valueon Z axis'].replace(out1,lower,inplace=True)
X['Time domain:Bodyacceleration jerk,mean valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,mean valueon Z axis'])

# sns.boxplot(X['Time domain:Bodyacceleration jerk,mean valueon Z axis'])
# plt.show()

# # sns.boxplot(X['Time domain:Bodyacceleration,mean valueon X axis'])
# # plt.show()
# print(X['Time domain:Bodyacceleration,mean valueon X axis'])
# Q1 = X['Time domain:Bodyacceleration,mean valueon X axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration,mean valueon X axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration,mean valueon X axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration,mean valueon X axis'] > upper].values
# X['Time domain:Bodyacceleration,mean valueon X axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration,mean valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration,mean valueon X axis'])
# sns.boxplot(X['Time domain:Bodyacceleration,mean valueon X axis'])
# matplotlib.pyplot.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration,mean valueon Y axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration,mean valueon Y axis'])
# Q1 = X['Time domain:Bodyacceleration,mean valueon Y axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration,mean valueon Y axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration,mean valueon Y axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration,mean valueon Y axis'] > upper].values
# X['Time domain:Bodyacceleration,mean valueon Y axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration,mean valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration,mean valueon Y axis'])
# sns.boxplot(X['Time domain:Bodyacceleration,mean valueon Y axis'])
# plt.show()

#
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration,mean valueon Z axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration,mean valueon Z axis'])
# Q1 = X['Time domain:Bodyacceleration,mean valueon Z axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration,mean valueon Z axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration,mean valueon Z axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration,mean valueon Z axis'] > upper].values
# X['Time domain:Bodyacceleration,mean valueon Z axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration,mean valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration,mean valueon Z axis'])
# sns.boxplot(X['Time domain:Bodyacceleration,mean valueon Z axis'])
# plt.show()


# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration,standard deviration valueon X axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration,standard deviration valueon X axis'])
# Q1 = X['Time domain:Bodyacceleration,standard deviration valueon X axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration,standard deviration valueon X axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration,standard deviration valueon X axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration,standard deviration valueon X axis'] > upper].values
# X['Time domain:Bodyacceleration,standard deviration valueon X axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration,standard deviration valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration,standard deviration valueon X axis'])
# sns.boxplot(X['Time domain:Bodyacceleration,standard deviration valueon X axis'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'])
# Q1 = X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'] > upper].values
# X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'])
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon X axis'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'])
# Q1 = X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'] > upper].values
# X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'])
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon Y axis'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'])
# plt.show()
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'])
# Q1 = X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'] < lower].values
# out2 = X[X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'] > upper].values
# X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'])
# sns.boxplot(X['Time domain:Bodyacceleration jerk,standard deviration valueon Z axis'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyangular velocity jerk,mean valueon X axis'])
# plt.show()
# print(X['Time domain:Bodyangular velocity jerk,mean valueon X axis'])
# Q1 = X['Time domain:Bodyangular velocity jerk,mean valueon X axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyangular velocity jerk,mean valueon X axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyangular velocity jerk,mean valueon X axis'] < lower].values
# out2 = X[X['Time domain:Bodyangular velocity jerk,mean valueon X axis'] > upper].values
# X['Time domain:Bodyangular velocity jerk,mean valueon X axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyangular velocity jerk,mean valueon X axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyangular velocity jerk,mean valueon X axis'])
# sns.boxplot(X['Time domain:Bodyangular velocity jerk,mean valueon X axis'])
# plt.show()

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'])
# plt.show()
# print(X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'])
# Q1 = X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'].quantile(0.25)
# Q3 = X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# upper = Q3 + 1.5*IQR
# lower = Q1 - 1.5*IQR
# print(upper)
# print(lower)
# out1 = X[X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'] < lower].values
# out2 = X[X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'] > upper].values
# X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'].replace(out1,lower,inplace=True)
# X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'].replace(out2,upper,inplace=True)
# print(X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'])
# sns.boxplot(X['Time domain:Bodyangular velocity jerk,mean valueon Y axis'])
# plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ran = RandomForestClassifier()


X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,random_state=0,test_size=0.3)
ran.fit(X_train,Y_train)

Y_pred1 = ran.predict(X_test)
print(accuracy_score(Y_test,Y_pred1)*100)