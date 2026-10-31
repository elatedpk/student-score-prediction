import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [12, 25, 32, 40, 50, 55, 65, 72, 80, 90]
}

df = pd.DataFrame(data)

x = df[['Hours_Studied']]
y = df['Scores']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Predicted Scores:", y_pred)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

hours = float(input("\nEnter the number of hours studied: "))

user_input = [[hours]]
predicted_score = model.predict(user_input)

print("Predicted Score:", round(predicted_score[0], 2))
