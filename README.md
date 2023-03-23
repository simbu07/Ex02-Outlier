# Ex02-Outlier
## AIM:
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
## Explanation
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Get the information about the data

### STEP 3
Detect the Outliers using IQR method and Z score

### STEP 4
Remove the outliers

### STEP 5
Plot the datas using Box Plot

## PROGRAM:
```
Developed By : Silambarasan K
Reg No : 212221230101
```
### 1 & 2 Examine price_per_sqft column and use IQR to remove outliers and create new dataframe:
```python
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 - Data Science/bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)

df.head()

q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_per_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
```
### 3 Examine price_per_sqft column and use zscore of 3 to remove outliers.:
```
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
```
### 4-(i) For the data set height_weight.csv detect weight outliers using IQR method:

```python
df3 = pd.read_csv("height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape

sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
```
 ### 4-(ii) For the data set height_weight.csv detect height outliers using IQR method:
 ```python
 sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
 ```
## OUTPUT:
### 1 & 2 Examine price_per_sqft column and use IQR to remove outliers and create new dataframe:
### Dataset:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/dataset.png)
### Dataset Head:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/head.png)
### Dataset Info:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/info.png)
### Dataset Describe:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/describe.png)
### Null Values:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/isnull.png)
### Dataset Shape:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/shape.png)

### Box plot of price_per_sqft column with outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/with_outliers.png)
### price_per_sqft - Dataset after removing outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/dataset_without_outliers.png)
### price_per_sqft - Shape of Dataset after removing outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/dataset_shape_without_outliers.png)
### Box Plot of price_per_sqft column without outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/bhp/without_outliers.png)
## (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
### Dataset after removal of outlier using z score:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/zscore/afterremoval.png)
### Shape of Dataset after removal of outlier using z score:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/zscore/shape.png)
### price_per_sqft column after removing outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/zscore/after_removal.png)
## (4) For the data set height_weight.csv detect weight and height outliers using IQR method: 

### Dataset:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/df.png)
### Dataset Head:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/head.png)
### Dataset Info:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/info.png)
### Dataset Describe:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/describe.png)
### Null Values:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/isnull.png)
### Dataset Shape:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/shape.png)
### Weight - With outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/weight_with_outlier.png)
### Weight - Dataset after removing Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/weight_iqr.png)
### Weight - Shape of Dataset after removing Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/weight_without_outlier_shape.png)
### Weight - Without Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/weight_without_outlier.png)
### Height - With outliers:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/height_iqr.png)
### Height - Dataset after removing Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/height_with_outlier.png)

### Height - Shape of Dataset after removing Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/height_without_outlier_shape.png)
### Height - Without Outliers using IQR method:
![dataset](https://github.com/ShafeeqAhamedS/Ex02-Outlier/blob/main/img/height_weight/height_without_outlier.png)

## RESULT:
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.


