import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LarsCV, ElasticNet, ElasticNetCV, LinearRegression, LassoCV

# import statsmodels.api as sm
# from pandas_profiling import ProfileReport

df = pd.read_csv('./data/Admission_Prediction.csv')

df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())
df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mode()[0])

x = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = df[['Chance of Admit']]

sc = StandardScaler()
data_x = sc.fit_transform(x)
x_std = pd.DataFrame(data_x, columns=x.columns)

X_train, X_test, y_train, y_test = train_test_split(x_std, y, random_state=42, test_size=0.30)

elasticnet_cv = ElasticNetCV(cv=20)
elasticnet_cv.fit(X_train, y_train)

elastic_lr = ElasticNet(alpha=elasticnet_cv.alpha_,
                        l1_ratio=elasticnet_cv.l1_ratio,
                        random_state=42)
print(elastic_lr.fit(X_train, y_train))
print(elastic_lr.score(X_test, y_test))

#pickle.dump(elastic_lr, open("elastic.pickle", 'wb'))

filename = 'elastic.pickle'  # 'finalized_model.pickle'
x = [[317.0, 245.0, 4.0, 2.0, 3.0, 4.2, 1.0]]
print(type(x))
df_x = pd.DataFrame(x)
sc = StandardScaler()
data_x = sc.fit_transform(df_x)
data_x = pd.DataFrame(data_x)
loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
# predictions using the loaded model file
prediction = loaded_model.predict(data_x)
print('prediction is', prediction)
