# evaluation.py
# Evaluation system for the RAG application

import re
import logging
from typing import List, Dict, Any
from datetime import datetime
import statistics
import json

logger = logging.getLogger(__name__)

class EvaluationDataset:
    """Contains evaluation questions and expected answers for GitHub API"""
    
    def __init__(self):
        self.qa_pairs = [
            {
                "question": "How do I authenticate with the GitHub API?",
                "ground_truth": "You can authenticate with GitHub API using Personal Access Tokens (PAT) in the Authorization header, OAuth apps, or GitHub Apps. The most common method is using a PAT with 'Authorization: token YOUR_TOKEN' header.",
                "expected_keywords": ["personal access token", "authorization", "header", "oauth", "authentication", "token"]
            },
            {
                "question": "How can I list repositories for a user?",
                "ground_truth": "Use GET /users/{username}/repos endpoint to list public repositories for a user. You can use query parameters like per_page, page, type, and sort to customize the results.",
                "expected_keywords": ["GET", "/users/", "repos", "endpoint", "per_page", "page", "type", "sort"]
            },
            {
                "question": "What are the rate limits for GitHub API?",
                "ground_truth": "GitHub API has different rate limits: 5000 requests per hour for authenticated requests, 60 for unauthenticated. Rate limit info is included in response headers like X-RateLimit-Limit and X-RateLimit-Remaining.",
                "expected_keywords": ["rate limit", "5000", "60", "authenticated", "unauthenticated", "X-RateLimit", "headers"]
            },
            {
                "question": "How do I create a repository using the API?",
                "ground_truth": "Use POST /user/repos endpoint to create a repository. Send JSON payload with name (required), description, private (boolean), and other optional parameters. Requires repo scope for PAT.",
                "expected_keywords": ["POST", "/user/repos", "endpoint", "JSON", "name", "description", "private", "repo scope"]
            },
            {
                "question": "What should I do if I get a 404 error?",
                "ground_truth": "404 errors typically mean the resource doesn't exist or you don't have permission to access it. Check the URL, ensure you have proper authentication, and verify you have the required scopes or permissions.",
                "expected_keywords": ["404", "resource", "permission", "authentication", "scopes", "access"]
            },
            {
                "question": "How do webhooks work in GitHub?",
                "ground_truth": "GitHub webhooks send HTTP POST requests to configured URLs when specific events occur in repositories. You can configure webhooks in repository settings or via the API using POST /repos/{owner}/{repo}/hooks.",
                "expected_keywords": ["webhooks", "HTTP POST", "events", "repositories", "POST", "/repos/", "hooks", "configure"]
            },
            {
                "question": "How can I search for repositories?",
                "ground_truth": "Use GET /search/repositories endpoint with query parameters. You can search by name, description, topics, language, stars, forks, and more. Use 'q' parameter for the search query.",
                "expected_keywords": ["GET", "/search/repositories", "query", "parameters", "name", "language", "stars", "forks"]
            },
            {
                "question": "What are the different types of GitHub tokens?",
                "ground_truth": "GitHub has Personal Access Tokens (PAT), OAuth app tokens, GitHub App tokens, and installation tokens. Each has different scopes and use cases. PATs are user-specific, while app tokens are for applications.",
                "expected_keywords": ["personal access token", "oauth", "github app", "installation", "tokens", "scopes", "applications"]
            }
        ]
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """Return all Q&A pairs for evaluation"""
        return self.qa_pairs
    
    def add_qa_pair(self, question: str, ground_truth: str, expected_keywords: List[str]):
        """Add a new Q&A pair to the dataset"""
        self.qa_pairs.append({
            "question": question,
            "ground_truth": ground_truth,
            "expected_keywords": expected_keywords
        })

class RAGEvaluator:
    """Evaluates RAG system responses against ground truth"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
    
    def calculate_f1_metrics(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate F1, precision, and recall based on word overlap"""
        try:
            # Tokenize and normalize
            predicted_words = set(re.findall(r'\b\w+\b', predicted.lower()))
            ground_truth_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
            
            # Calculate metrics
            if not predicted_words and not ground_truth_words:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            
            if not predicted_words:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            if not ground_truth_words:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # Calculate intersection
            intersection = predicted_words.intersection(ground_truth_words)
            
            precision = len(intersection) / len(predicted_words) if predicted_words else 0.0
            recall = len(intersection) / len(ground_truth_words) if ground_truth_words else 0.0
            
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        except Exception as e:
            logger.error(f"Error calculating F1 metrics: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def calculate_rouge_scores(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        try:
            # Simple ROUGE-1 implementation (unigram overlap)
            predicted_words = re.findall(r'\b\w+\b', predicted.lower())
            ground_truth_words = re.findall(r'\b\w+\b', ground_truth.lower())
            
            if not predicted_words or not ground_truth_words:
                return {"rouge1_f": 0.0, "rouge1_p": 0.0, "rouge1_r": 0.0}
            
            # ROUGE-1
            predicted_set = set(predicted_words)
            ground_truth_set = set(ground_truth_words)
            
            overlap = predicted_set.intersection(ground_truth_set)
            
            rouge1_precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
            rouge1_recall = len(overlap) / len(ground_truth_set) if ground_truth_set else 0.0
            rouge1_f = (2 * rouge1_precision * rouge1_recall) / (rouge1_precision + rouge1_recall) if (rouge1_precision + rouge1_recall) > 0 else 0.0
            
            return {
                "rouge1_f": rouge1_f,
                "rouge1_p": rouge1_precision,
                "rouge1_r": rouge1_recall
            }
        
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"rouge1_f": 0.0, "rouge1_p": 0.0, "rouge1_r": 0.0}
    
    def calculate_keyword_coverage(self, predicted: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Calculate how many expected keywords are present in the predicted answer"""
        try:
            predicted_lower = predicted.lower()
            found_keywords = []
            
            for keyword in expected_keywords:
                if keyword.lower() in predicted_lower:
                    found_keywords.append(keyword)
            
            coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
            
            return {
                "coverage": coverage,
                "found_keywords": found_keywords,
                "expected_keywords": expected_keywords,
                "found_count": len(found_keywords),
                "expected_count": len(expected_keywords)
            }
        
        except Exception as e:
            logger.error(f"Error calculating keyword coverage: {e}")
            return {
                "coverage": 0.0,
                "found_keywords": [],
                "expected_keywords": expected_keywords,
                "found_count": 0,
                "expected_count": len(expected_keywords)
            }
    
    def evaluate_response(self, question: str, predicted_response: str, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of a single response"""
        try:
            ground_truth = qa_pair["ground_truth"]
            expected_keywords = qa_pair.get("expected_keywords", [])
            
            # Calculate all metrics
            f1_metrics = self.calculate_f1_metrics(predicted_response, ground_truth)
            rouge_scores = self.calculate_rouge_scores(predicted_response, ground_truth)
            keyword_results = self.calculate_keyword_coverage(predicted_response, expected_keywords)
            
            # Overall quality score (weighted combination)
            quality_score = (
                f1_metrics["f1"] * 0.4 +
                rouge_scores["rouge1_f"] * 0.3 +
                keyword_results["coverage"] * 0.3
            )
            
            return {
                "question": question,
                "predicted_response": predicted_response,
                "ground_truth": ground_truth,
                "f1_metrics": f1_metrics,
                "rouge_scores": rouge_scores,
                "keyword_coverage": keyword_results["coverage"],
                "found_keywords": keyword_results["found_keywords"],
                "expected_keywords": keyword_results["expected_keywords"],
                "quality_score": quality_score,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error evaluating response for question '{question}': {e}")
            return {
                "question": question,
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def batch_evaluate(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate multiple Q&A pairs and return aggregate results"""
        try:
            results = []
            
            for qa_pair in qa_pairs:
                question = qa_pair["question"]
                
                # Get response from RAG system
                if self.rag_system:
                    rag_response = self.rag_system.get_response(question)
                    predicted_response = rag_response["answer"]
                else:
                    predicted_response = "No RAG system available"
                
                # Evaluate response
                evaluation_result = self.evaluate_response(question, predicted_response, qa_pair)
                results.append(evaluation_result)
            
            # Calculate aggregate metrics
            valid_results = [r for r in results if "error" not in r]
            
            if valid_results:
                f1_scores = [r["f1_metrics"]["f1"] for r in valid_results]
                rouge_scores = [r["rouge_scores"]["rouge1_f"] for r in valid_results]
                keyword_coverages = [r["keyword_coverage"] for r in valid_results]
                quality_scores = [r["quality_score"] for r in valid_results]
                
                aggregate_metrics = {
                    "f1_score": {
                        "mean": statistics.mean(f1_scores),
                        "std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                        "min": min(f1_scores),
                        "max": max(f1_scores)
                    },
                    "rouge1_f": {
                        "mean": statistics.mean(rouge_scores),
                        "std": statistics.stdev(rouge_scores) if len(rouge_scores) > 1 else 0.0,
                        "min": min(rouge_scores),
                        "max": max(rouge_scores)
                    },
                    "keyword_coverage": {
                        "mean": statistics.mean(keyword_coverages),
                        "std": statistics.stdev(keyword_coverages) if len(keyword_coverages) > 1 else 0.0,
                        "min": min(keyword_coverages),
                        "max": max(keyword_coverages)
                    },
                    "quality_score": {
                        "mean": statistics.mean(quality_scores),
                        "std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                        "min": min(quality_scores),
                        "max": max(quality_scores)
                    }
                }
                
                return {
                    "individual_results": results,
                    "aggregate_metrics": aggregate_metrics,
                    "summary": {
                        "total_questions": len(qa_pairs),
                        "successful_evaluations": len(valid_results),
                        "failed_evaluations": len(results) - len(valid_results),
                        "overall_quality_score": aggregate_metrics["quality_score"]["mean"],
                        "evaluation_timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return {
                    "individual_results": results,
                    "aggregate_metrics": {},
                    "summary": {
                        "total_questions": len(qa_pairs),
                        "successful_evaluations": 0,
                        "failed_evaluations": len(results),
                        "overall_quality_score": 0.0,
                        "evaluation_timestamp": datetime.now().isoformat()
                    }
                }
        
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return {
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], output_file: str = None) -> str:
        """Generate a detailed evaluation report"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("RAG SYSTEM EVALUATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Summary section
            summary = evaluation_results.get("summary", {})
            report_lines.append("SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Questions: {summary.get('total_questions', 0)}")
            report_lines.append(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}")
            report_lines.append(f"Failed Evaluations: {summary.get('failed_evaluations', 0)}")
            report_lines.append(f"Overall Quality Score: {summary.get('overall_quality_score', 0.0):.3f}")
            report_lines.append("")
            
            # Aggregate metrics
            aggregate = evaluation_results.get("aggregate_metrics", {})
            if aggregate:
                report_lines.append("AGGREGATE METRICS")
                report_lines.append("-" * 40)
                
                for metric_name, metric_data in aggregate.items():
                    report_lines.append(f"{metric_name.upper().replace('_', ' ')}:")
                    report_lines.append(f"  Mean: {metric_data.get('mean', 0.0):.3f}")
                    report_lines.append(f"  Std:  {metric_data.get('std', 0.0):.3f}")
                    report_lines.append(f"  Min:  {metric_data.get('min', 0.0):.3f}")
                    report_lines.append(f"  Max:  {metric_data.get('max', 0.0):.3f}")
                    report_lines.append("")
            
            # Individual results
            individual_results = evaluation_results.get("individual_results", [])
            if individual_results:
                report_lines.append("INDIVIDUAL QUESTION RESULTS")
                report_lines.append("-" * 40)
                
                for i, result in enumerate(individual_results, 1):
                    if "error" in result:
                        report_lines.append(f"Q{i}: {result['question']}")
                        report_lines.append(f"ERROR: {result['error']}")
                        report_lines.append("")
                        continue
                    
                    report_lines.append(f"Q{i}: {result['question']}")
                    report_lines.append(f"Quality Score: {result.get('quality_score', 0.0):.3f}")
                    report_lines.append(f"F1 Score: {result.get('f1_metrics', {}).get('f1', 0.0):.3f}")
                    report_lines.append(f"ROUGE-1 F: {result.get('rouge_scores', {}).get('rouge1_f', 0.0):.3f}")
                    report_lines.append(f"Keyword Coverage: {result.get('keyword_coverage', 0.0):.3f}")
                    
                    found_kw = result.get('found_keywords', [])
                    expected_kw = result.get('expected_keywords', [])
                    missing_kw = [kw for kw in expected_kw if kw not in found_kw]
                    
                    report_lines.append(f"Found Keywords: {', '.join(found_kw) if found_kw else 'None'}")
                    report_lines.append(f"Missing Keywords: {', '.join(missing_kw) if missing_kw else 'None'}")
                    report_lines.append("")
            
            report_text = "\n".join(report_lines)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Evaluation report saved to {output_file}")
            
            return report_text
        
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return f"Error generating report: {str(e)}"
    
    def save_results_json(self, evaluation_results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}")


class EvaluationRunner:
    """Main class to run RAG system evaluations"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.dataset = EvaluationDataset()
        self.evaluator = RAGEvaluator(rag_system)
    
    def run_full_evaluation(self, save_report: bool = True, report_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        try:
            logger.info("Starting RAG system evaluation...")
            
            # Get evaluation dataset
            qa_pairs = self.dataset.get_qa_pairs()
            logger.info(f"Loaded {len(qa_pairs)} evaluation questions")
            
            # Run batch evaluation
            results = self.evaluator.batch_evaluate(qa_pairs)
            
            if save_report:
                import os
                os.makedirs(report_dir, exist_ok=True)
                
                # Generate timestamp for file names
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save JSON results
                json_file = os.path.join(report_dir, f"evaluation_results_{timestamp}.json")
                self.evaluator.save_results_json(results, json_file)
                
                # Generate and save text report
                report_file = os.path.join(report_dir, f"evaluation_report_{timestamp}.txt")
                report_text = self.evaluator.generate_evaluation_report(results, report_file)
                
                logger.info(f"Evaluation completed. Results saved to {report_dir}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            return {"error": str(e)}
    
    def add_custom_question(self, question: str, ground_truth: str, expected_keywords: List[str]):
        """Add a custom evaluation question"""
        self.dataset.add_qa_pair(question, ground_truth, expected_keywords)
        logger.info(f"Added custom question: {question}")
    
    def evaluate_single_question(self, question: str, ground_truth: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate a single question"""
        try:
            # Get response from RAG system
            if self.rag_system:
                rag_response = self.rag_system.get_response(question)
                predicted_response = rag_response["answer"]
            else:
                predicted_response = "No RAG system available"
            
            # Create temporary QA pair
            qa_pair = {
                "question": question,
                "ground_truth": ground_truth,
                "expected_keywords": expected_keywords
            }
            
            # Evaluate
            result = self.evaluator.evaluate_response(question, predicted_response, qa_pair)
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating single question: {e}")
            return {"error": str(e)}


# Example usage and testing functions
def main():
    """Example usage of the evaluation system"""
    try:
        # Initialize evaluation runner (without RAG system for demo)
        runner = EvaluationRunner()
        
        # Run full evaluation
        results = runner.run_full_evaluation()
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print(f"Evaluation completed!")
            print(f"Total questions: {summary['total_questions']}")
            print(f"Overall quality score: {summary['overall_quality_score']:.3f}")
        else:
            print("Evaluation failed:", results.get("error", "Unknown error"))
    
    except Exception as e:
        print(f"Error running evaluation: {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()