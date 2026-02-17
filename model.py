import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Load dataset
data = pd.read_csv("data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
