import sys
print(sys.executable)
"C:/Users/hp/Downloads/Anaconda3-2024.10-1-Windows-x86_64.exe"
"C:/Users/hp/Downloads/Anaconda3-2024.10-1-Windows-x86_64.exe"
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

upload = widgets.FileUpload(accept='.csv', multiple=False)
display(upload)

def get_uploaded_file():
    if upload.value:
        uploaded_filename = list(upload.value.keys())[0]
        content = upload.value['mental-and-substance-use-as-share-of-disease']['content']
        import io
        return pd.read_csv(io.BytesIO(content))
    else:
        print("mental-and-substance-use-as-share-of-disease")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
df1=pd.read_csv("C:/Users/hp/Downloads/mental-and-substance-use-as-share-of-disease.csv")

df2=pd.read_csv("C:/Users/hp/Downloads/mental-and-substance-use-as-share-of-disease.csv")
df1.head()
df2.head(10)
data=pd.merge(df1,df2)
data.head(10)
data.isnull().sum()
data.drop('Code',axis=1,inplace=True)
data.head(10)
data.size,data.shape
# Rename the columns by assigning the result of set_axis back to data
# Removing inplace=True as it's no longer supported
# Also, update the list of column names to match the actual number of columns (3) after dropping 'Code'
data = data.set_axis(['Entity','Year','DALYs'], axis='columns')

# Verify the column names have been updated
print(data.head(10))
# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(12,6))
# Calculate correlation on the numeric data
sns.heatmap(numeric_data.corr(),annot=True,cmap='Blues')
plt.plot()
sns.pairplot(data,corner=True)
plt.show()
mean=data['DALYs'].mean()
mean
fig=px.pie(data,values='DALYs',names='Entity',title='Mental Health')
fig.show()
fig=px.line(data,x='Year',y='DALYs',color='Entity',title='Mental Health')
fig.show()
data.info()
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in data.columns:
  if data[i].dtype == 'object':
    data[i]=l.fit_transform(data[i])
data.shape
x=data.drop('DALYs',axis=1)
y=data['DALYs']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print("xtrain:",x_train.shape)
print("xtest:",x_test.shape)
print("ytrain:",y_train.shape)
print("ytest:",y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lr=LinearRegression()
lr.fit(x_train,y_train)
ytrain_pred=lr.predict(x_test)
mse=mean_squared_error(y_test,ytrain_pred)
rmse=(np.sqrt(mean_squared_error(y_test,ytrain_pred)))
r2=r2_score(y_test,ytrain_pred)
print("The Linear Regression model performance for training set")
print("--------------------------------------------------------")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
ytrain_pred=rf.predict(x_test)
mse=mean_squared_error(y_test,ytrain_pred)
rmse=(np.sqrt(mean_squared_error(y_test,ytrain_pred)))
r2=r2_score(y_test,ytrain_pred)
print("The Random Forest Regression model performance for training set")
print("--------------------------------------------------------")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
ytest_pred=lr.predict(x_test)
mse=mean_squared_error(y_test,ytest_pred)
rmse=(np.sqrt(mean_squared_error(y_test,ytest_pred)))
r2=r2_score(y_test,ytest_pred)
print("The Linear Regression model performance for testing set")
print("--------------------------------------------------------")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
ytest_pred=rf.predict(x_test)
mse=mean_squared_error(y_test,ytest_pred)
rmse=(np.sqrt(mean_squared_error(y_test,ytest_pred)))
r2=r2_score(y_test,ytest_pred)
print("The Random Forest Regression model performance for testing set")
print("--------------------------------------------------------")
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
