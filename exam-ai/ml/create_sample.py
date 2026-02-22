import pandas as pd
df = pd.read_csv("data/Questions.csv", encoding="latin-1")
df = df[['Title', 'Body', 'Score']].dropna()
df_sample = df.sample(10000, random_state=42)

df_sample.to_csv("data/sample_questions.csv", index=False)

print("Sample dataset created: data/sample_questions.csv")