import pandas as pd
import numpy as np

def generate_dataset(num_students=100, num_questions=15, max_marks=5):

    students = [f"S{i}" for i in range(1, num_students + 1)]
    questions = [f"Q{i}" for i in range(1, num_questions + 1)]

    data = []

    for student in students:
        for question in questions:
            marks = np.random.randint(0, max_marks + 1)
            data.append([question, student, marks])

    df = pd.DataFrame(data, columns=["question", "student_id", "marks"])

    df.to_csv("student_responses.csv", index=False)

    print("Dataset generated successfully!")

if __name__ == "__main__":
    generate_dataset()