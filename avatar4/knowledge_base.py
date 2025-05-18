import random
import time
import re
from typing import List, Dict, Any

def generate_question(knowledge_base: Dict[str, Any], domain: str, difficulty: str) -> Dict[str, Any]:
    """
    Dynamically generates a question based on the domain and difficulty
    with an anti-repetition mechanism and more diversity.

    Args:
        knowledge_base: The knowledge base dictionary.
        domain: The technical domain.
        difficulty: The difficulty level.

    Returns:
        A dictionary containing the generated question and its metadata.
    """
    if domain not in knowledge_base:
        return _get_fallback_question(domain, difficulty)

    domain_knowledge = knowledge_base[domain]

    # Select an appropriate concept for the difficulty level
    # Check the history to avoid repetitions
    if not hasattr(generate_question, 'used_concepts'):
        generate_question.used_concepts = set()

    # Filter out already used concepts
    available_concepts = []
    if difficulty in domain_knowledge["difficulty_levels"]:
        available_concepts = [c for c in domain_knowledge["difficulty_levels"][difficulty]
                              if c not in generate_question.used_concepts]

    # If all concepts have been used, reset
    if not available_concepts:
        generate_question.used_concepts = set()
        available_concepts = domain_knowledge["difficulty_levels"][difficulty]
    else:
        available_concepts = [c for c in domain_knowledge["concepts"]
                              if c not in generate_question.used_concepts]

    # Select a new concept
    concept = random.choice(available_concepts)
    generate_question.used_concepts.add(concept)  # Mark as used

    # Select a related concept for comparison questions
    related_concepts = [c for c in domain_knowledge["concepts"]
                         if c != concept and c not in generate_question.used_concepts]

    if not related_concepts:  # If all used, allow reuse for comparisons
        related_concepts = [c for c in domain_knowledge["concepts"] if c != concept]

    related_concept = random.choice(related_concepts) if related_concepts else concept

    # Question templates in English
    question_templates = [
        "Can you explain the concept of {concept}?",
        "How would you use {concept} in a concrete case?",
        "What is the difference between {concept} and {related_concept}?",
        "Could you give a practical example of using {concept}?",
        "What are the advantages and disadvantages of {concept}?",
        "How would you implement {concept} in a real project?",
        "In what context is {concept} particularly useful?",
        "Can you compare the effectiveness of {concept} compared to {related_concept}?",
        "How would you optimize an implementation of {concept}?",
        "What problems does {concept} solve effectively?",
        "How would you explain {concept} to a junior developer?",
        "What are the pitfalls to avoid when using {concept}?",
        "How does {concept} integrate into the {domain} ecosystem?",
        "What alternatives to {concept} do you know?",
        "How would you test an implementation of {concept}?"
    ]

    # Use different question formats depending on the difficulty
    if difficulty == "easy":
        template_subset = question_templates[:7]  # Simpler questions
    elif difficulty == "medium":
        template_subset = question_templates[3:12]  # Intermediate questions
    else:  # hard
        template_subset = question_templates[5:]  # More complex questions

    # Select a template and generate the question
    template = random.choice(template_subset)
    question = template.format(concept=concept, related_concept=related_concept)

    # Extract keywords from the concept for evaluation
    keywords = _extract_keywords(concept) + _extract_keywords(related_concept)

    # Add a situational context to the question to make it more concrete
    question = _add_context_to_question(question, domain, concept)

    return {
        "id": f"{domain}_{difficulty}_{int(time.time())}",
        "question": question,
        "difficulty": difficulty,
        "concept": concept,
        "related_concept": related_concept,
        "expected_keywords": keywords,
        "follow_ups": _generate_follow_ups(domain, concept, related_concept)
    }

def _add_context_to_question(question: str, domain: str, concept: str) -> str:
    """
    Adds a situational context to the question to make it more concrete.
    """
    # Do not add context to questions that already have a specific format
    if "difference between" in question or "compare" in question:
        return question

    contexts = {
        "python": [
            "In a test automation project,",
            "When developing a REST API,",
            "For a data analysis script,",
            "In a Django web application,"
        ],
        "machine learning": [
            "For a sales prediction project,",
            "In a recommendation system,",
            "For a computer vision application,",
            "In a sentiment analysis project,"
        ],
        "algorithms": [
            "To optimize the performance of an application,",
            "In a database management system,",
            "For a network routing algorithm,",
            "In a search engine,"
        ],
        "databases": [
            "In a high-load application,",
            "For a real-time reporting system,",
            "In a microservices architecture,"
        ],
        "web_development": [
            "For a responsive SPA application,",
            "In a secure authentication system,",
            "For a high-performance user interface,"
        ]
    }

    # Get contexts for the domain or use generic contexts
    domain_contexts = contexts.get(domain, [
        "In a professional context,",
        "For a critical project in a company,",
        "In a high-availability application,"
    ])

    # Add a scenario if the question lends itself to it
    if "how" in question.lower() or "explain" in question.lower() or "give an example" in question.lower():
        return f"{random.choice(domain_contexts)} {question}"

    return question

def _extract_keywords(concept: str) -> List[str]:
    """Extracts keywords from a concept for evaluating responses."""
    # Simplification: divides the concept into words and filters out short words
    return [word for word in re.split(r'\W+', concept.lower()) if len(word) > 3]

def _generate_follow_ups(domain: str, concept: str, related_concept: str) -> List[str]:
    """Generates follow-up questions based on the main concept."""
    follow_up_templates = [
        f"Can you give a concrete example of using {concept}?",
        f"What are common pitfalls to avoid with {concept}?",
        f"How could you optimize or improve {concept}?",
        f"In what context would you prefer to use {related_concept} rather than {concept}?",
        f"How would you implement {concept} in a production environment?",
        f"What are the technical challenges in implementing {concept}?",
        f"How is {concept} evolving in recent versions of {domain}?",
        f"What is your personal experience with {concept}?",
        f"How would you debug a problem related to {concept}?",
        f"How would you explain {concept} to a non-technical colleague?",
        f"What metrics would you use to evaluate the effectiveness of {concept}?",
        f"How does {concept} integrate with other technologies or frameworks?"
    ]

    # Randomly select 2-3 follow-up questions
    return random.sample(follow_up_templates, min(3, len(follow_up_templates)))

def _get_fallback_question(domain: str, difficulty: str) -> Dict[str, Any]:
    """Provides a fallback question if the domain is not in the knowledge base."""
    fallback_questions = {
        "easy": f"Can you explain a basic concept in {domain}?",
        "medium": f"Describe a common challenge in {domain} and how you would address it.",
        "hard": f"Explain an advanced concept in {domain} and its application in a professional context."
    }

    question = fallback_questions.get(difficulty, fallback_questions["medium"])

    return {
        "id": f"{domain}_{difficulty}_{int(time.time())}",
        "question": question,
        "difficulty": difficulty,
        "concept": "general concept",
        "related_concept": domain,
        "expected_keywords": [domain.lower()],
        "follow_ups": [
            f"Could you elaborate further on {domain}?",
            f"How have you applied these concepts in your previous projects?"
        ]
    }