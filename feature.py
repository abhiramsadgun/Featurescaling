import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv')
print(df.head())
print(df.columns)
df.drop(['month','day','year'],axis=1,inplace=True)
print(df.head())
df['Classes']=np.where(df['Classes'].str.contains("not fire"),0,1)
print(df.head())
print(df.tail())
print(df['Classes'].value_counts())
x=df.drop('FWI',axis=1)
y=df['FWI']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.corr())

#multi-correaltion
plt.figure(figsize=(12,10))
coor=x_train.corr()
sns.heatmap(coor,cmap="coolwarm")
plt.show()

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                cloname=corr_matrix.columns[i]
                col_corr.add(cloname)
    return col_corr
dc=correlation(x_train,0.85)
#drop the features more than the threshold
x_train.drop(dc,axis=1,inplace=True)
x_test.drop(dc,axis=1,inplace=True)

#preprocessing

from sklearn.preprocessing import StandardScaler

scal=StandardScaler()
x_train_scale=scal.fit_transform(x_train)
x_test_scale=scal.transform(x_test)
print(x_train_scale)

#boxplot
plt.subplots(figsize=(15,5))
plt.subplots(2,12)
sns.boxplot(data=x_train)
plt.title('x_tain before scaling')
plt.show()
plt.subplots(2,12)
sns.boxplot(data=x_train_scale)
plt.title('x_tain after scaling')
plt.show() 