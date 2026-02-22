from performance_analysis import analyze_exam

result, insights = analyze_exam("../student_responses.csv")

print("\n=== Question Analysis ===")
print(result)

print("\n=== Insights ===")
print(insights)
from visualization import (
    plot_question_performance,
    plot_quality_distribution,
    plot_score_distribution
)
from performance_analysis import load_data



df = load_data("../student_responses.csv")

print("\nShowing visualizations...")

plot_question_performance(result)
plot_quality_distribution(result)
plot_score_distribution(df)