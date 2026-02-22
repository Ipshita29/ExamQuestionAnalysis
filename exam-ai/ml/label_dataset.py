import pandas as pd

df = pd.read_csv("data/sample_questions.csv")

# temporary defaults for missing columns
df['AnswerCount'] = 0
df['ViewCount'] = 0

def assign_difficulty(row):
    score = row['Score']
    answers = row['AnswerCount']
    views = row['ViewCount']

    # Improved logic (multi-feature)
    if score >= 10 or (answers >= 3 and views >= 1000):
        return "Easy"

    elif score >= 3 or (answers >= 1 and views >= 300):
        return "Medium"

    else:
        return "Hard"

df['Difficulty'] = df.apply(assign_difficulty, axis=1)

df.to_csv("data/labeled_questions.csv", index=False)
print("Improved difficulty labels created → data/labeled_questions.csv")