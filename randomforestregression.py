#random forest regression
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values


#splitting the data set into dataset and training set
"""from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)"""


#feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#fitting the Random Forest Regression model to the dataset
#create your regressor here
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,Y)
#Predicting a new result with  Random Forest Regression
Y_pred=regressor.predict([[6.5]])

#Visualising the  Random Forest Regression results for higher regression and smoother curve
#Visualising Polynomial Regression Results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("True of Bluff( Random Forest Regression Model )")
plt.xlabel("PositionLevel")
plt.ylabel("Salary")
plt.show()

