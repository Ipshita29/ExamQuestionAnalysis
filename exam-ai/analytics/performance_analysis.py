import pandas as pd
import numpy as np
import joblib
import os


# -------------------------------
# Load Dataset + Validation
# -------------------------------
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


# -------------------------------
# Compute Metrics
# -------------------------------
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


# -------------------------------
# Question Quality Classification
# -------------------------------
def detect_question_quality(pass_rate, discrimination):
    quality = {}

    for q in pass_rate.index:
        pr = pass_rate[q]
        di = discrimination[q]

        if pr > 0.85:
            quality[q] = "Too Easy"
        elif pr < 0.30:
            quality[q] = "Too Hard"
        elif di < 0.15:
            quality[q] = "Confusing / Poor Discrimination"
        elif di >= 0.40:
            quality[q] = "Excellent"
        else:
            quality[q] = "Acceptable"

    return pd.Series(quality)


# -------------------------------
# Student Ranking
# -------------------------------
def compute_student_ranking(df):
    total_scores = df.groupby("student_id")["marks"].sum()
    return total_scores.sort_values(ascending=False)


# -------------------------------
# Learning Gap Detection
# -------------------------------
def detect_learning_gaps(pass_rate, discrimination):
    weak_questions = []

    for q in pass_rate.index:
        if pass_rate[q] < 0.5 or discrimination[q] < 0.2:
            weak_questions.append(q)

    return weak_questions


# -------------------------------
# Exam Summary (JSON-safe types)
# -------------------------------
def generate_exam_summary(result_df):
    total_questions = len(result_df)

    summary = {
        "total_questions": int(total_questions),
        "excellent_questions": int((result_df["quality"] == "Excellent").sum()),
        "too_easy": int((result_df["quality"] == "Too Easy").sum()),
        "too_hard": int((result_df["quality"] == "Too Hard").sum()),
        "confusing": int((result_df["quality"].str.contains("Confusing")).sum()),
    }

    summary["good_percentage"] = float(
        round(100 * summary["excellent_questions"] / total_questions, 2)
    )

    return summary


# -------------------------------
# Teacher Report
# -------------------------------
def generate_teacher_report(summary, weak_questions):
    report = [
        f"Total Questions: {summary['total_questions']}",
        f"Excellent Questions: {summary['excellent_questions']}",
        f"Too Easy Questions: {summary['too_easy']}",
        f"Too Hard Questions: {summary['too_hard']}",
        f"Confusing Questions: {summary['confusing']}",
    ]

    if weak_questions:
        report.append(f"Learning gaps detected in: {', '.join(weak_questions)}")
    else:
        report.append("No major learning gaps detected.")

    return "\n".join(report)


# -------------------------------
# ML Integration (Safe)
# -------------------------------
def load_ml_components(model_path, vectorizer_path):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("ML model not found → skipping ML integration")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_ml_difficulty(question_texts, model, vectorizer):
    if model is None or vectorizer is None:
        return ["ML Not Available"] * len(question_texts)

    X = vectorizer.transform(question_texts)
    return model.predict(X)


# -------------------------------
# MAIN ANALYTICS ENGINE
# -------------------------------
def analyze_exam(path):
    df = load_data(path)

    avg_score = compute_avg_score(df)
    pass_rate = compute_pass_rate(df)
    discrimination = compute_discrimination_index(df)
    quality = detect_question_quality(pass_rate, discrimination)
    ranking = compute_student_ranking(df)
    learning_gaps = detect_learning_gaps(pass_rate, discrimination)

    result = pd.DataFrame({
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "discrimination_index": discrimination,
        "quality": quality
    })

    # ---------------- ML Integration ----------------
    model_path = "../ml/models/difficulty_model.pkl"
    vectorizer_path = "../ml/models/vectorizer.pkl"

    model, vectorizer = load_ml_components(model_path, vectorizer_path)

    question_texts = result.index.astype(str).tolist()
    ml_predictions = predict_ml_difficulty(question_texts, model, vectorizer)

    result["ml_difficulty"] = ml_predictions


    summary = generate_exam_summary(result)
    teacher_report = generate_teacher_report(summary, learning_gaps)

    insights = {
        "exam_summary": summary,
        "weak_questions": learning_gaps,
        "top_students": ranking.head(5).index.tolist(),
        "bottom_students": ranking.tail(5).index.tolist(),
        "teacher_report": teacher_report
    }

    return result, insights