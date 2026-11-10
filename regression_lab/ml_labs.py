#pip install pandas numpy scikit-learn matplotlib seaborn
#install required packages 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)
print("All libraries loaded successfully!")

data = {
    "area_sqft": [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600],
    "bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
    "age_years": [20, 15, 12, 10, 8, 7, 6, 5, 4, 3],
    "price": [160000, 195000, 235000, 270000, 305000,
              345000, 380000, 415000, 465000, 500000]
}

df = pd.DataFrame(data)
print(df)
# 4. Define Features and Target
X = df[["area_sqft", "bedrooms", "age_years"]]
y = df["price"]
print(X)
print(y)

# 5. Split Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  #Without random_state, your accuracy will change every run

# 6. Train the Regression Model
model = LinearRegression()
model.fit(X_train, y_train)   #The model learns from training data.
y_pred = model.predict(X_test)   #The model answers questions using what it learned.
print("Predictions:", y_pred)

# 7. Evaluate Regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# 8. Predict a New House
new_house = pd.DataFrame({
    "area_sqft": [1900],
    "bedrooms": [4],
    "age_years": [5]
})

print("Predicted price:", model.predict(new_house)[0])
'''
data = {"area_sqt":[1500, 1200, 2000, 4000],
        "home": [2, 1, 3, 5],
        "years":[3, 2, 4, 1],
        "price": [20000, 15000, 25000, 50000]}
d = pd.DataFrame(data)
print(d)   

x = d[["area_sqt","home","years"]]
y = d["price"]
print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Prediction:",y_pred)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rsme = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(mae,mse,rsme,r2)


n = pd.DataFrame({"area_sqt":[1400],
                  "home":[2],
                  "years":[2]})

print(model.predict(n)) '''
