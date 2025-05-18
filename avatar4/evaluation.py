import re
from typing import Dict, Any
import logging

def evaluate_response(question_data: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Evaluates a candidate's response to a technical question.

    Args:
        question_data: The question data, including expected keywords.
        response: The response provided by the candidate.

    Returns:
        A dictionary containing the score, feedback, and recognized keywords.
    """
    if not response:
        return {
            "score": 0.0,
            "feedback": "I didn't understand your answer. Can you repeat it, please?",
            "needs_clarification": True
        }

    # Lowercase for better matching
    response = response.lower()
    expected_keywords = [k.lower() for k in question_data["expected_keywords"]]

    # Count present keywords
    matched_keywords = [k for k in expected_keywords if k in response]
    keyword_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0

    # Analyze the length and structure of the response
    words = response.split()
    length_score = min(len(words) / 50.0, 1.0)  # Normalize to 1.0 max

    # Detect examples of code or technical terms
    has_code = bool(re.search(r'[{}\[\]()\/]', response))
    has_technical = bool(re.search(r'\b(function|class|method|algorithm|data|code)\b', response))

    # Calculate the final score
    technical_score = 0.3 if has_technical else 0
    code_score = 0.2 if has_code else 0
    final_score = (keyword_score * 0.5) + (length_score * 0.2) + technical_score + code_score

    # Generate feedback
    feedback = []
    if final_score >= 0.8:
        feedback.append("Your answer is complete and well-structured.")
        if has_code:
            feedback.append("Good use of technical examples.")
    elif final_score >= 0.6:
        feedback.append("Your answer shows good understanding.")
        if not has_technical:
            feedback.append("Try to include more technical details.")
    elif final_score >= 0.4:
        feedback.append("Your answer covers some important points.")
        feedback.append("Consider developing the technical aspects further.")
    else:
        feedback.append("Your answer lacks precision.")
        feedback.append("Try adding concrete examples and technical concepts.")

    # Check if clarification is needed
    needs_clarification = (
        final_score < 0.3 or
        "what do you mean" in response or
        "can you clarify" in response or
        "i don't understand" in response or
        "can you explain" in response
    )

    return {
        "score": final_score,
        "feedback": " ".join(feedback),
        "matched_keywords": matched_keywords,
        "needs_clarification": needs_clarification
    }