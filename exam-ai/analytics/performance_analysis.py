import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    required_cols = {"question", "student_id", "marks"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Dataset must contain: question, student_id, marks")

    if df.empty:
        raise ValueError("Dataset is empty")

    df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
    df = df.dropna(subset=["marks"])

    return df


def compute_avg_score(df):
    return df.groupby("question")["marks"].mean()


def compute_pass_rate(df, pass_marks=2):
    df = df.copy()
    df["pass"] = df["marks"] >= pass_marks
    return df.groupby("question")["pass"].mean()

def compute_max_marks_per_question(df):
    return df.groupby("question")["marks"].max()

def compute_discrimination_index(df):
    student_scores = df.groupby("student_id")["marks"].sum().sort_values()

    n = len(student_scores)
    k = max(1, int(0.27 * n)) 

    bottom_students = student_scores.head(k).index
    top_students = student_scores.tail(k).index

    max_marks_per_q = compute_max_marks_per_question(df)
    di_values = {}

    for q in df["question"].unique():
        q_data = df[df["question"] == q]

        top_avg = q_data[q_data["student_id"].isin(top_students)]["marks"].mean()
        bottom_avg = q_data[q_data["student_id"].isin(bottom_students)]["marks"].mean()

        max_marks = max_marks_per_q[q]

        if pd.isna(top_avg) or pd.isna(bottom_avg) or max_marks == 0:
            di = 0
        else:
            di = (top_avg - bottom_avg) / max_marks

        di_values[q] = di

    return pd.Series(di_values)


def detect_question_quality(pass_rate, discrimination):
    quality = {}

    for q in pass_rate.index:
        pr = pass_rate[q]
        di = discrimination[q]

        if pr > 0.85 and di < 0.20:
            quality[q] = "Too Easy (Low Discrimination)"
        elif pr < 0.30 and di < 0.20:
            quality[q] = "Too Hard / Confusing"
        elif di >= 0.40:
            quality[q] = "Excellent Question"
        elif di >= 0.20:
            quality[q] = "Good Question"
        else:
            quality[q] = "Poor Question"

    return pd.Series(quality)


def compute_student_ranking(df):
    total_scores = df.groupby("student_id")["marks"].sum()
    ranking = total_scores.sort_values(ascending=False)
    return ranking

def detect_learning_gaps(avg_score, threshold=0.4):
    max_score = avg_score.max()
    weak_questions = avg_score[avg_score < threshold * max_score]
    return weak_questions.index.tolist()


def analyze_exam(path):
    df = load_data(path)

    avg_score = compute_avg_score(df)
    pass_rate = compute_pass_rate(df)
    discrimination = compute_discrimination_index(df)
    quality = detect_question_quality(pass_rate, discrimination)
    ranking = compute_student_ranking(df)
    learning_gaps = detect_learning_gaps(avg_score)

    result = pd.DataFrame({
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "discrimination_index": discrimination,
        "quality": quality
    })

    insights = {
        "weak_questions": learning_gaps,
        "top_students": ranking.head(5).index.tolist(),
        "bottom_students": ranking.tail(5).index.tolist()
    }

    return result, insights