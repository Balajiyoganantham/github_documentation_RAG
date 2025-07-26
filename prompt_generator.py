# prompt_generator.py
# Handles different prompting methods for the GitHub API RAG system

import re
from typing import Dict, Callable

class PromptGenerator:
    """Different prompting methods for GitHub API assistance"""
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text.strip())
        return len(words)
    
    @staticmethod
    def validate_question(question: str) -> tuple[bool, str]:
        """Validate question length and content"""
        if not question or not question.strip():
            return False, "Question content is required"
        word_count = PromptGenerator.count_words(question)
        if word_count < 3:
            return False, f"Question too short ({word_count} words). Minimum 3 words required."
        if word_count > 100:
            return False, f"Question too long ({word_count} words). Maximum 100 words allowed."
        return True, f"Question length valid ({word_count} words)"
    
    @staticmethod
    def create_chain_of_thoughts_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""I need to answer this GitHub API question step by step. Let me think through this systematically:

Step 1: First, I'll understand what the user is trying to accomplish
Step 2: Then, I'll identify the relevant GitHub API concepts and endpoints
Step 3: Next, I'll examine the authentication and authorization requirements
Step 4: After that, I'll understand the request/response patterns
Step 5: Finally, I'll provide practical implementation guidance

Let me work through each step:

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Now, let me create a comprehensive answer following my step-by-step analysis:

Please provide a clear, actionable response that includes authentication requirements, API endpoints, request examples, and practical implementation steps based on my systematic analysis above."""

    @staticmethod
    def create_tree_of_thoughts_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""I will analyze this GitHub API question using multiple reasoning paths and then synthesize the best approach:

Path 1: Authentication & Security Approach
- Focus on proper authentication methods
- Emphasize security best practices and token management

Path 2: API Design & Implementation Approach  
- Focus on RESTful principles and HTTP methods
- Emphasize practical code examples and patterns

Path 3: Error Handling & Best Practices Approach
- Focus on common pitfalls and error scenarios
- Emphasize robust implementation strategies

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Now, synthesizing insights from all three reasoning paths, provide a comprehensive answer that covers authentication, API usage, error handling, and best practices."""

    @staticmethod
    def create_role_based_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""You are Zoro, a senior GitHub API expert with 10 years of experience in API integration and developer advocacy. Your expertise lies in quickly identifying the right API endpoints and providing clear, actionable guidance for developers.

As an expert GitHub API consultant, your task is to:
1. Quickly identify the user's specific GitHub API need
2. Determine the appropriate authentication method required
3. Select the correct API endpoints and HTTP methods
4. Provide practical code examples and implementation steps
5. Address potential challenges and best practices

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Based on your expert analysis, provide a comprehensive answer that includes authentication requirements, API endpoints, code examples, and best practices in a clear, developer-friendly tone."""

    @staticmethod
    def create_react_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""I'll use the ReAct approach (Reasoning + Acting) to answer this GitHub API question:

Thought 1: I need to understand what the user wants to accomplish
Action 1: Analyze the question and identify the core GitHub API functionality needed
Observation 1: [After analysis] I understand the user's goal

Thought 2: I should identify the appropriate API endpoints
Action 2: Search through the documentation for relevant endpoints
Observation 2: [After search] I found the relevant API endpoints

Thought 3: I need to determine authentication requirements
Action 3: Check what authentication method is needed for these endpoints
Observation 3: [After check] I understand the authentication requirements

Thought 4: I should provide practical implementation steps
Action 4: Create code examples and step-by-step guidance
Observation 4: [After creation] I have practical implementation guidance

Thought 5: I need to address potential issues and best practices
Action 5: Identify common pitfalls and provide best practices
Observation 5: [After analysis] I can provide comprehensive guidance

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Based on my ReAct analysis above, provide a comprehensive answer covering authentication, API endpoints, implementation steps, and best practices."""

    @staticmethod
    def create_directional_stimulus_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""Focus your analysis on creating a structured GitHub API response that developers would find immediately actionable and valuable. Your answer should demonstrate deep understanding of GitHub API patterns and practical implementation.

Key directions for your analysis:
→ Authentication: What authentication method is required and how to set it up?
→ Endpoints: Which specific API endpoints should be used?
→ Implementation: What are the exact HTTP requests and responses?
→ Examples: How can this be implemented in code?
→ Best Practices: What are the important considerations and potential issues?

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Following these directions, create a comprehensive answer that would satisfy developer standards for GitHub API integration and implementation."""

    @staticmethod
    def create_step_back_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""Before diving into the specific API details, let me step back and consider the bigger picture:

High-level question: What is the fundamental GitHub API concept or pattern the user needs to understand?
Broader context: How does this specific need fit into the larger landscape of GitHub API capabilities?
Meta-question: What makes this API usage important and what are the key principles involved?

Now, with this broader perspective in mind, let me analyze the specific implementation details:

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Using both the high-level perspective and detailed analysis, provide a comprehensive answer that covers the fundamental concepts, specific implementation, and practical guidance while maintaining awareness of the broader GitHub API ecosystem."""

    @staticmethod
    def create_zero_shot_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""Answer the following GitHub API question using the provided documentation context. Your response should include authentication requirements, API endpoints, implementation examples, and best practices. Write in a clear, developer-friendly style.

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}

Provide your answer now."""

    @staticmethod
    def create_one_shot_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""Here's an example of how to answer a GitHub API question:

Example Question: "How do I create a repository using the GitHub API?"
Example Answer: "To create a repository, you need to authenticate with a Personal Access Token (PAT) that has the 'repo' scope. Send a POST request to https://api.github.com/user/repos with the Authorization header: 'Authorization: token YOUR_PAT'. The request body should be JSON with required 'name' field and optional fields like 'description' and 'private'. Example: POST /user/repos with body {{'name': 'my-repo', 'description': 'My new repository', 'private': true}}. The response will be a Repository object with details about the created repository."

Now, following the same format and style, answer this GitHub API question:

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}"""

    @staticmethod
    def create_few_shot_prompt(question: str, context: str, chat_history: str = "") -> str:
        return f"""Here are examples of how to answer GitHub API questions:

Example 1:
Question: "How do I authenticate with the GitHub API?"
Answer: "GitHub API supports several authentication methods. For scripts and command-line tools, use Personal Access Tokens (PATs). Include the PAT in the Authorization header: 'Authorization: token YOUR_PAT'. For OAuth apps, use the OAuth flow to get an access token. Always treat tokens like passwords and grant only necessary scopes. For public data, you can make unauthenticated requests but with rate limits."

Example 2:
Question: "How do I list repositories for a user?"
Answer: "Use GET /users/{username}/repos for public repositories or GET /user/repos for authenticated user's repositories. Include authentication for private repos. Optional query parameters: 'type' (all, owner, member), 'sort' (created, updated, pushed, full_name), 'direction' (asc, desc), 'per_page' (max 100). Example: GET /user/repos?type=owner&sort=updated&per_page=10"

Now, following the same format and style as these examples, answer this GitHub API question:

User Question: {question}

Available Documentation Context:
{context}

Chat History: {chat_history}"""

    @classmethod
    def create_prompt_by_method(cls, question: str, context: str, method: str, chat_history: str = "") -> str:
        prompt_methods: Dict[str, Callable[[str, str, str], str]] = {
            'chain_of_thoughts': cls.create_chain_of_thoughts_prompt,
            'tree_of_thoughts': cls.create_tree_of_thoughts_prompt,
            'role_based': cls.create_role_based_prompt,
            'react': cls.create_react_prompt,
            'directional_stimulus': cls.create_directional_stimulus_prompt,
            'step_back': cls.create_step_back_prompt,
            'zero_shot': cls.create_zero_shot_prompt,
            'one_shot': cls.create_one_shot_prompt,
            'few_shot': cls.create_few_shot_prompt
        }
        
        if method in prompt_methods:
            return prompt_methods[method](question, context, chat_history)
        else:
            return cls.create_zero_shot_prompt(question, context, chat_history)  # Default fallback

# Available prompting methods
PROMPTING_METHODS = {
    'chain_of_thoughts': 'Chain-of-Thoughts',
    'tree_of_thoughts': 'Tree-of-Thoughts', 
    'role_based': 'Role-based prompting',
    'react': 'ReAct prompting',
    'directional_stimulus': 'Directional Stimulus prompting',
    'step_back': 'Step-Back prompting',
    'zero_shot': 'Zero-shot',
    'one_shot': 'One-shot',
    'few_shot': 'Few-shot'
} 