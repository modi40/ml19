import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("weather.csv")

features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df[features])
y = df['Play'].map({'No': 0, 'Yes': 1})

nb = CategoricalNB()
nb.fit(X_encoded, y)

print("Naive Bayes Training Accuracy:", accuracy_score(y, nb.predict(X_encoded)))

new_weather = pd.DataFrame({
    'Outlook': ['Overcast'],
    'Temperature': ['Mild'],
    'Humidity': ['Normal'],
    'Windy': [False]
})
pred = nb.predict(encoder.transform(new_weather))[0]
result = "Yes (Play)" if pred == 1 else "No (Do Not Play)"
print(f"Prediction for weather [Overcast, Mild, Normal, False]: {result}")
