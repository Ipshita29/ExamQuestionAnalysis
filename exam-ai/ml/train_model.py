import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("./exam-ai/data/labeled_questions.csv")
df["text"] = df["Title"] + " " + df["Body"]

X = df["text"]
y = df["Difficulty"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight="balanced",max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "./exam-ai/models/difficulty_model.pkl")
joblib.dump(vectorizer, "./exam-ai/models/vectorizer.pkl")

print("Model saved!")