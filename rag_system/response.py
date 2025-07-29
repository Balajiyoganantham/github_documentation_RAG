from datetime import datetime
import logging

def calculate_confidence(question, answer, source_documents):
    try:
        confidence = 0.0
        # Factor 1: Number of source documents (0-0.3)
        doc_count = len(source_documents)
        if doc_count >= 5:
            confidence += 0.3
        elif doc_count >= 3:
            confidence += 0.2
        elif doc_count >= 1:
            confidence += 0.1
        # Factor 2: Answer length and detail (0-0.2)
        if len(answer) > 200 and "I don't" not in answer:
            confidence += 0.2
        elif len(answer) > 100:
            confidence += 0.1
        # Factor 3: Specific API terms present (0-0.3)
        api_terms = ['GET ', 'POST ', 'PUT ', 'DELETE ', '/api/', 'endpoint', 'parameter', 'header', 'token']
        api_terms_found = sum(1 for term in api_terms if term.lower() in answer.lower())
        confidence += min(0.3, api_terms_found * 0.05)
        # Factor 4: No uncertainty phrases (0-0.2)
        uncertainty_phrases = ["I don't know", "I'm not sure", "I don't have", "unclear"]
        if not any(phrase.lower() in answer.lower() for phrase in uncertainty_phrases):
            confidence += 0.2
        return min(1.0, confidence)
    except Exception:
        return 0.5 