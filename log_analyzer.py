import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("logs.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Convert text into ML features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])

# Labels
y = df['severity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train classification model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Anomaly Detection
anomaly_model = IsolationForest(contamination=0.2)
anomaly_model.fit(X.toarray())

df['anomaly'] = anomaly_model.predict(X.toarray())

print("\nAnomaly Detection Results:\n")
print(df[['message', 'anomaly']])

# Recommendation Engine
recommendations = {
    "database connection timeout":
    "Check database connection pool",

    "memory usage high":
    "Restart memory-intensive services",

    "disk write failure":
    "Check disk health and free space",

    "cpu overheating detected":
    "Check cooling system and CPU load"
}

print("\nRecommendations:\n")

for msg in df['clean_message']:
    for key in recommendations:
        if key in msg:
            print(f"{msg} --> {recommendations[key]}")