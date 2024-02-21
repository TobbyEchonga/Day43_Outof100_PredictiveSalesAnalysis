import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample sales data (replace with your own dataset)
data = {
    'Date': pd.date_range('2022-01-01', periods=12, freq='M'),
    'Sales': [100, 120, 130, 110, 150, 160, 170, 140, 180, 200, 210, 190]
}

df = pd.DataFrame(data)

# Feature engineering: Extracting month and year from the date
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Month', 'Year']], df['Sales'], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], marker='o', linestyle='-', color='b', label='Actual Sales')
plt.plot(X_test['Month'] + X_test['Year'] * 12, predictions, marker='o', linestyle='--', color='r', label='Predicted Sales')
plt.title('Predictive Sales Analysis')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
