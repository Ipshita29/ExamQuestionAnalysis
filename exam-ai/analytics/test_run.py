from performance_analysis import analyze_exam

result, insights = analyze_exam("../student_responses.csv")

print("\n=== Question Analysis ===")
print(result)

print("\n=== Insights ===")
print(insights)