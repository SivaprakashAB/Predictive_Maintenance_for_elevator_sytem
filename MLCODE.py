import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def predict_elevator_breakdown(data, temperature, vibration, current_usage, current_hour, current_day_of_week):
    # Feature Engineering
    data['Date'] = pd.to_datetime(data['Date'])
    data['Hour'] = data['Date'].dt.hour
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data.drop(['Date', 'Breakdown'], axis=1, inplace=True)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Splitting the data into features (X) and target (y)
    X = data.drop('BreakdownWithin12Hrs', axis=1)
    y = data['BreakdownWithin12Hrs']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=999)
    model.fit(X_train, y_train)

    # Predicting on user input
    user_input = pd.DataFrame({
        'Temperature': [temperature],
        'Vibration': [vibration],
        'CurrentUsage': [current_usage],
        'Hour': [current_hour],
        'DayOfWeek': [current_day_of_week]
    })
    prediction = model.predict(user_input)[0]

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return prediction, accuracy, precision, recall , X , y , model

# Load dataset
data = pd.read_csv('elevator_data.csv')

# Take user input for prediction
temperature = float(input("Enter temperature: "))
vibration = float(input("Enter vibration: "))
current_usage = float(input("Enter current usage: "))
current_hour = pd.to_datetime('now').hour  # Include current hour
current_day_of_week = pd.to_datetime('now').dayofweek  # Include current day of week

# Make prediction
prediction, accuracy, precision, recall ,X ,y,model = predict_elevator_breakdown(data, temperature, vibration, current_usage, current_hour, current_day_of_week)

# Print prediction
if prediction == 1:
    print("The elevator is predicted to break down within a month.")
else:
    print("The elevator is predicted to NOT break down within a month.")

# Print accuracy, precision, and recall
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(X['Temperature'], y, color='blue', label='Actual Breakdown Within month') 
plt.scatter(temperature, prediction , color='red', marker='x',s=100, label='Predicted Breakdown Within month') 
plt.xlabel('Temperature')
plt.ylabel('BreakdownWithinmonth')
plt.legend()
plt.title('Actual vs Predicted Breakdown Within month')
plt.show()