# Car Price Prediction based on vehicle age and mileage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Data set 
data = pd.DataFrame({
    "car_age_years": [1, 2, 4, 6, 8],
    "mileage_km":[5000, 15000, 30000, 50000, 80000],
    "price": [25,20,15,10,5]
})

print("------dataset--------")
print (data)

X = data[["car_age_years" , "mileage_km"]]
Y =  data["price"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Model 

model = LinearRegression()
model.fit(X_train, Y_train)
#Model accuracy
accuracy = model.score(X_test, Y_test)
print(f"\n=== Model Accuracy: {accuracy*100:.1f}% ===")

print("\n=== Predict Your Result ===")
car_age_years = int(input("car_age_years: "))
mileage_km = int(input("mileage_km: "))
result = model.predict([[car_age_years, mileage_km]])
print(result)