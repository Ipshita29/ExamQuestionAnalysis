import json
import tempfile
import os
from langchain_core.tools import tool
from analytics.performance_analysis import analyze_exam


@tool
def run_analysis_tool(csv_content: str) -> str:
    """
    Runs full statistical exam analysis on a CSV of student responses.
    The CSV must have columns: question, student_id, marks.
    Returns a JSON string with per-question stats and overall insights.
    If data is noisy or incomplete the tool returns a structured error message.
    """
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        result_df, insights = analyze_exam(tmp_path)

        per_question = {}
        for q in result_df.index:
            row = result_df.loc[q]
            per_question[str(q)] = {
                "avg_score": round(float(row["avg_score"]), 3),
                "pass_rate": round(float(row["pass_rate"]), 3),
                "discrimination_index": round(float(row["discrimination_index"]), 3),
                "quality": str(row["quality"]),
                "ml_difficulty": str(row["ml_difficulty"]),
            }

        output = {
            "per_question_stats": per_question,
            "exam_summary": insights["exam_summary"],
            "weak_questions": insights["weak_questions"],
            "top_students": insights["top_students"],
            "bottom_students": insights["bottom_students"],
            "teacher_report": insights["teacher_report"],
        }
        return json.dumps(output)

    except ValueError as exc:
        return json.dumps({"error": str(exc), "status": "invalid_data"})
    except Exception as exc:
        return json.dumps({"error": f"Unexpected analysis failure: {exc}", "status": "error"})
    finally:
        if tmp and os.path.exists(tmp_path):
            os.unlink(tmp_path)
