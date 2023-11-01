# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries,read the data set.find the number of null data.
2. Find the number of null data.
3. Import sklearn library.
4. Find y predict and plot the graph.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: karnan k
RegisterNumber: 212222230062 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:
### data.head():
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/a6c14921-1331-499b-b87c-73c0f1246ed4)

### data.info():
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/ab81013a-dd41-4fda-b233-7e7c6fbb4ddd)

### data.isnull().sum():
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/bb49a839-66d5-4802-98b9-44bf7bcba946)

### Elbow method:
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/dd9666ae-e04c-4598-b03f-24215ae0a8a1)

### K-Means:
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/726fb3cf-dded-4071-95db-cf7839635375)

### Array value of Y:
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/82c00b13-16c7-45c8-864d-884df44c308f)

### Customer Segmentation:
![image](https://github.com/karnankasinathan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787064/daa90fac-ab0c-4b7b-ac67-45d1577413e6)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
