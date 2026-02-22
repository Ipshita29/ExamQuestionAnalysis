import matplotlib.pyplot as plt
import seaborn as sns

def plot_question_performance(result_df):
    plt.figure(figsize=(10, 5))
    result_df["avg_score"].plot(kind="bar")
    plt.title("Average Score per Question")
    plt.xlabel("Question")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.show()



def plot_quality_distribution(result_df):
    plt.figure(figsize=(7, 4))
    result_df["quality"].value_counts().plot(kind="bar")
    plt.title("Question Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()



def plot_score_distribution(df):
    plt.figure(figsize=(7, 4))
    sns.histplot(df["marks"], bins=10, kde=True)
    plt.title("Overall Score Distribution")
    plt.xlabel("Marks")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()