import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_question_performance(result_df):
    plt.figure(figsize=(12, 6))

    result_df_sorted = result_df.sort_index(
        key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
    )

    colors = result_df_sorted["quality"].map({
        "Excellent": "green",
        "Acceptable": "blue",
        "Confusing / Poor Discrimination": "red",
        "Too Easy": "orange",
        "Too Hard": "purple"
    })

    result_df_sorted["avg_score"].plot(kind="bar", color=colors)

    plt.title("Average Score per Question (Colored by Quality)", fontsize=14)
    plt.xlabel("Question")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_quality_distribution(result_df):
    plt.figure(figsize=(8, 5))

    quality_counts = result_df["quality"].value_counts()

    sns.barplot(x=quality_counts.index, y=quality_counts.values)

    plt.title("Question Quality Distribution", fontsize=14)
    plt.xlabel("Quality Category")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def plot_score_distribution(df):
    plt.figure(figsize=(8, 5))

    sns.histplot(df["marks"], bins=10, kde=True)

    plt.title("Overall Student Score Distribution", fontsize=14)
    plt.xlabel("Marks")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_di_vs_pass(result_df):
    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=result_df["pass_rate"],
        y=result_df["discrimination_index"],
        hue=result_df["quality"],
        palette="deep",
        s=100
    )

    plt.axhline(0.2, linestyle="--", color="gray")
    plt.axvline(0.5, linestyle="--", color="gray")

    plt.title("Pass Rate vs Discrimination Index", fontsize=14)
    plt.xlabel("Pass Rate")
    plt.ylabel("Discrimination Index")
    plt.legend(title="Quality")
    plt.tight_layout()
    plt.show()