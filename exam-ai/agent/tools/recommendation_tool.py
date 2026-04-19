import json
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from config import settings


class QuestionRecommendation(BaseModel):
    question_id: str = Field(description="The question identifier")
    current_quality: str = Field(description="Current quality label from analysis")
    issue: str = Field(description="Concise description of the problem with this question")
    recommendation: str = Field(description="Specific, actionable improvement recommendation")
    bloom_level_suggestion: str = Field(
        description="Suggested Bloom's taxonomy level to target"
    )
    priority: str = Field(description="Priority level: High, Medium, or Low")


class AssessmentRecommendationReport(BaseModel):
    overall_verdict: str = Field(description="1–2 sentence overall assessment of the exam quality")
    key_learning_gaps: list[str] = Field(
        description="List of topic areas / questions with significant learning gaps"
    )
    per_question_recommendations: list[QuestionRecommendation] = Field(
        description="Structured recommendations for each question needing improvement"
    )
    immediate_actions: list[str] = Field(
        description="Top 3 immediate actions the instructor should take"
    )
    long_term_suggestions: list[str] = Field(
        description="2–3 strategic changes for future assessment design"
    )


@tool
def generate_recommendations_tool(analysis_json: str, pedagogy_context: str) -> str:
    """
    Generates a structured, actionable assessment improvement report.
    Takes a JSON string of analysis results and a string of retrieved pedagogical context.
    Returns a JSON string containing structured improvement recommendations for the exam.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )
        structured_llm = llm.with_structured_output(AssessmentRecommendationReport)

        prompt = f"""You are an expert educational assessor reviewing an exam's statistical analysis.

ANALYSIS RESULTS:
{analysis_json}

RELEVANT PEDAGOGICAL GUIDANCE:
{pedagogy_context}

Based on the analysis data and pedagogical best practices above, generate a comprehensive,
structured assessment improvement report. Focus on actionable, specific recommendations grounded
in the data. For questions labeled as 'Too Easy', 'Too Hard', or 'Confusing / Poor Discrimination',
provide targeted improvement strategies. Identify learning gaps from questions with low pass rates
or discrimination indices. Be concise but specific."""

        report: AssessmentRecommendationReport = structured_llm.invoke(prompt)
        return json.dumps(report.model_dump())

    except Exception as exc:
        fallback = {
            "error": f"Recommendation generation failed: {exc}",
            "overall_verdict": "Unable to generate AI recommendations. Please review the raw analysis data.",
            "key_learning_gaps": [],
            "per_question_recommendations": [],
            "immediate_actions": ["Review questions with pass_rate < 0.50", "Review questions with discrimination_index < 0.20"],
            "long_term_suggestions": ["Consult the full analysis report for details"],
        }
        return json.dumps(fallback)
