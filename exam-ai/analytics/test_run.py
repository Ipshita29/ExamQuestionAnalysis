from performance_analysis import analyze_exam, load_data
from visualization import (
    plot_question_performance,
    plot_quality_distribution,
    plot_score_distribution,
    plot_di_vs_pass
)

result, insights = analyze_exam("../student_responses.csv")

print("\n=== Question Analysis ===")
print(result)

print("\n=== Insights ===")
print(insights)


df = load_data("../student_responses.csv")

print("\nShowing visualizations...")

plot_question_performance(result)
plot_quality_distribution(result)
plot_score_distribution(df)
plot_di_vs_pass(result)