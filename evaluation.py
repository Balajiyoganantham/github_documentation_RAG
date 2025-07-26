# evaluation.py
# Handles evaluation logic for the RAG system

from datetime import datetime
from typing import List, Dict, Any
import string
import statistics
from rouge_score import rouge_scorer

class EvaluationDataset:
    """Manages evaluation dataset with ground truth Q&A pairs"""
    def __init__(self):
        self.qa_pairs = [
            {
                "question": "What is the base URL for all GitHub API requests?",
                "ground_truth": "All GitHub API requests should be made to https://api.github.com. It's recommended to include an Accept header to specify the desired media type, e.g., Accept: application/vnd.github.v3+json.",
                "keywords": ["base url", "https://api.github.com", "accept header", "media type", "v3", "api.github.com", "github api", "requests"]
            },
            {
                "question": "How do I authenticate with the GitHub API using a Personal Access Token?",
                "ground_truth": "Include the Personal Access Token (PAT) in the Authorization header: Authorization: token YOUR_PERSONAL_ACCESS_TOKEN. Treat PATs like passwords and grant only necessary scopes. PATs are alternative passwords for authentication with GitHub API.",
                "keywords": ["authentication", "personal access token", "pat", "authorization header", "token", "scopes", "security", "authorization", "alternative passwords", "github api"]
            },
            {
                "question": "Describe the OAuth flow for authenticating a GitHub API app.",
                "ground_truth": "The OAuth flow involves: 1) Redirecting the user to GitHub's authorization page, 2) User grants access, 3) GitHub redirects back with a code, 4) Exchange code for access_token, 5) Use access_token in the Authorization header. OAuth Apps are used for applications that need to act on behalf of users.",
                "keywords": ["oauth", "authorization", "access token", "redirect", "code", "flow", "access_token", "github authorization page", "client_id", "client_secret", "oauth apps"]
            },
            {
                "question": "How do you list repositories for the authenticated user?",
                "ground_truth": "Use GET /user/repos with authentication. Optional query parameters include type (all, owner, member), sort (created, updated, pushed, full_name), direction (asc, desc), per_page (max 100), and page. The response is an array of Repository objects.",
                "keywords": ["list repositories", "GET /user/repos", "authentication", "query parameters", "type", "sort", "direction", "per_page", "page", "/user/repos", "repository objects", "array"]
            },
            {
                "question": "How do you create a new repository for the authenticated user?",
                "ground_truth": "Send a POST request to /user/repos with authentication (public_repo or repo scope). The request body should include name (required), and optionally description, private, and auto_init. The response is the created Repository object with status code 201 Created.",
                "keywords": ["create repository", "POST", "/user/repos", "authentication", "repo scope", "name", "description", "private", "auto_init", "public_repo", "request body", "201 created", "repository object"]
            },
            {
                "question": "How do you list issues for a specific repository?",
                "ground_truth": "Use GET /repos/{owner}/{repo}/issues. Authentication is required for private repos. Optional query parameters include state (open, closed, all), creator, assignee, labels, sort (created, updated, comments), direction, since, per_page, and page.",
                "keywords": ["list issues", "GET /repos", "issues", "owner", "repo", "query parameters", "state", "labels", "per_page", "/repos/{owner}/{repo}/issues", "authentication", "private repos", "creator", "assignee", "sort", "direction", "since"]
            },
            {
                "question": "How do you create a new issue in a repository?",
                "ground_truth": "Send a POST request to /repos/{owner}/{repo}/issues with authentication (repo scope). The request body must include title, and can include body, milestone, labels, and assignees. The response is an Issue object with status code 201 Created.",
                "keywords": ["create issue", "POST", "/repos", "issues", "authentication", "repo scope", "title", "labels", "assignees", "/repos/{owner}/{repo}/issues", "request body", "body", "milestone", "issue object", "201 created"]
            },
            {
                "question": "How does pagination work in the GitHub API?",
                "ground_truth": "Use per_page and page query parameters to control pagination. The Link header in responses provides URLs for next, previous, first, and last pages. per_page specifies items per page (max 100, default 30), page specifies page number (default 1).",
                "keywords": ["pagination", "per_page", "page", "link header", "next", "last", "query parameters", "items per page", "page number", "link relations", "navigation", "urls"]
            },
            {
                "question": "What are some common HTTP status codes returned by the GitHub API and what do they mean?",
                "ground_truth": "Common status codes: 200 OK (success), 201 Created (resource created), 204 No Content (success, no content), 400 Bad Request, 401 Unauthorized, 403 Forbidden (or rate limit), 404 Not Found, 422 Unprocessable Entity, 500 Internal Server Error, 503 Service Unavailable.",
                "keywords": ["status code", "200", "201", "204", "400", "401", "403", "404", "422", "500", "503", "http status", "ok", "created", "unauthorized", "forbidden", "not found", "bad request", "unprocessable entity", "internal server error", "service unavailable"]
            },
            {
                "question": "What is the structure of an error response from the GitHub API?",
                "ground_truth": "Error responses include a message, an errors array with details, and a documentation_url. Example: { 'message': 'Validation Failed', 'errors': [...], 'documentation_url': '...' }. The errors array contains objects with resource, field, and code properties.",
                "keywords": ["error response", "message", "errors", "documentation_url", "validation failed", "error structure", "errors array", "resource", "field", "code", "json object"]
            },
            {
                "question": "How do you retrieve information about the authenticated user?",
                "ground_truth": "Use GET /user with authentication (PAT or OAuth token). The response is a User object with status code 200 OK. This endpoint requires authentication and returns information about the currently authenticated user.",
                "keywords": ["get user", "GET /user", "authentication", "user object", "/user", "pat", "oauth token", "200 ok", "authenticated user", "current user"]
            },
            {
                "question": "How do you retrieve information about a user by username?",
                "ground_truth": "Use GET /users/{username}. Authentication is optional but increases rate limits. The response is a User object. This endpoint retrieves public information about a specific user by their username.",
                "keywords": ["get user", "GET /users", "username", "user object", "authentication", "/users/{username}", "public information", "rate limits", "optional authentication", "specific user"]
            },
            {
                "question": "What are webhooks in GitHub and what are some common use cases?",
                "ground_truth": "Webhooks are HTTP callbacks that GitHub sends to your application when specific events occur. Common use cases: triggering CI/CD pipelines on push events, updating external issue trackers on issues events, sending notifications to chat applications (Slack, Discord, Teams), running security scans on new commits.",
                "keywords": ["webhooks", "events", "ci/cd", "push", "issue tracker", "notifications", "integrations", "subscribe", "http callbacks", "chat applications", "slack", "discord", "teams", "security scans", "automated workflows"]
            }
        ]
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        return self.qa_pairs

class RAGEvaluator:
    """Evaluation system for RAG pipeline using LangChain"""
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.evaluation_results = []
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better keyword matching"""
        text = text.lower().strip()
        # Remove punctuation but keep important symbols like /, -, :
        text = text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
        text = text.replace('{', ' ').replace('}', ' ').replace('"', ' ').replace("'", ' ')
        text = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
        return ' '.join(text.split())
    
    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with better handling of technical terms"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'use', 'used', 'using', 'get', 'gets', 'getting', 'send', 'sends', 'sending', 'make', 'makes', 'making',
            'include', 'includes', 'including', 'provide', 'provides', 'providing', 'require', 'requires', 'requiring',
            'need', 'needs', 'needing', 'want', 'wants', 'wanting', 'like', 'likes', 'liking', 'such', 'as', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else', 'while', 'until',
            'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
        
        # Extract keywords, keeping important technical terms
        keywords = []
        for word in words:
            if (word not in stop_words and 
                len(word) > 2 and 
                not word.isdigit() and
                not word.startswith('http')):
                keywords.append(word)
        
        # Add special handling for API endpoints and technical terms
        import re
        endpoints = re.findall(r'[a-z]+/[a-z0-9/{}]+', processed_text)
        keywords.extend(endpoints)
        
        # Add status codes
        status_codes = re.findall(r'\b(?:200|201|204|400|401|403|404|422|500|503)\b', processed_text)
        keywords.extend(status_codes)
        
        return list(set(keywords))  # Remove duplicates
    
    def calculate_f1_score(self, predicted_text: str, ground_truth_text: str) -> Dict[str, float]:
        """Enhanced F1 score calculation with better keyword matching"""
        predicted_keywords = set(self.extract_keywords(predicted_text))
        ground_truth_keywords = set(self.extract_keywords(ground_truth_text))
        
        if not ground_truth_keywords:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0, "predicted_keywords": [], "ground_truth_keywords": []}
        
        # Calculate intersection with partial matching for technical terms
        true_positives = 0
        matched_gt_keywords = set()
        
        for pred_keyword in predicted_keywords:
            for gt_keyword in ground_truth_keywords:
                # Exact match
                if pred_keyword == gt_keyword:
                    true_positives += 1
                    matched_gt_keywords.add(gt_keyword)
                    break
                # Partial match for technical terms
                elif (pred_keyword in gt_keyword or gt_keyword in pred_keyword) and len(pred_keyword) > 3:
                    true_positives += 0.8  # Partial credit
                    matched_gt_keywords.add(gt_keyword)
                    break
        
        precision = true_positives / len(predicted_keywords) if predicted_keywords else 0.0
        recall = len(matched_gt_keywords) / len(ground_truth_keywords) if ground_truth_keywords else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(true_positives),
            "predicted_keywords": list(predicted_keywords),
            "ground_truth_keywords": list(ground_truth_keywords)
        }
    
    def calculate_rouge_scores(self, predicted_text: str, ground_truth_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores for text similarity"""
        scores = self.rouge_scorer.score(ground_truth_text, predicted_text)
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge1_p": scores['rouge1'].precision,
            "rouge1_r": scores['rouge1'].recall,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rouge2_p": scores['rouge2'].precision,
            "rouge2_r": scores['rouge2'].recall,
            "rougeL_f": scores['rougeL'].fmeasure,
            "rougeL_p": scores['rougeL'].precision,
            "rougeL_r": scores['rougeL'].recall
        }
    
    def evaluate_response(self, question: str, predicted_response: str, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced evaluation with better keyword coverage calculation"""
        ground_truth_text = ground_truth_data["ground_truth"]
        
        # Calculate F1 metrics
        f1_metrics = self.calculate_f1_score(predicted_response, ground_truth_text)
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predicted_response, ground_truth_text)
        
        # Enhanced keyword coverage calculation
        expected_keywords = ground_truth_data.get("keywords", [])
        predicted_keywords = self.extract_keywords(predicted_response)
        
        # Calculate keyword coverage with partial matching
        matched_keywords = []
        for expected_keyword in expected_keywords:
            expected_lower = expected_keyword.lower()
            for pred_keyword in predicted_keywords:
                if (expected_lower == pred_keyword.lower() or 
                    expected_lower in pred_keyword.lower() or 
                    pred_keyword.lower() in expected_lower):
                    matched_keywords.append(expected_keyword)
                    break
        
        keyword_coverage = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        return {
            "question": question,
            "predicted_response": predicted_response,
            "ground_truth": ground_truth_text,
            "f1_metrics": f1_metrics,
            "rouge_scores": rouge_scores,
            "keyword_coverage": keyword_coverage,
            "expected_keywords": expected_keywords,
            "found_keywords": matched_keywords,
            "timestamp": datetime.now().isoformat()
        } 