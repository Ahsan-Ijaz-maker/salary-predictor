# CODE GIVE YOU ALLOWANCE TO PREDICT SALARIES BY EDUCATION LEVEL AND BY EXPERINECE 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#DATA SET 
data = pd.DataFrame({
    "experience": [1, 3, 5, 7, 10, 2],
    "education_level": [1, 2, 2, 3, 3, 1],
    "salary": [30000, 50000, 70000, 95000, 120000, 35000]
})

X = data[["experience", "education_level"]]
Y = data["salary"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#MODEL USED
model = LinearRegression()
model.fit(X_train, Y_train)
#ACCURACY CHECK 
accuracy = model.score(X_test, Y_test)
print(f"\n=== Model Accuracy: {accuracy*100:.1f}% ===")
#DATA INPUT 
print("\n=== Predict Your Result ===")
experience = int(input("Experience: "))
education_level = int(input("Education Level: "))
result = model.predict([[experience, education_level]])
print(result)