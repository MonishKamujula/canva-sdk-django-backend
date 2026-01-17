"""
Controllers for the cards app.

Business logic for generating presentation cards using OpenAI.
"""

import json
import logging
from typing import List, Optional

from openai import OpenAI

from core.ai import use_openai
from core.utils import replace_images
from .schemas import StepBreakdown, JsonOutput, CardList
from .prompts import STEP_PROMPT, get_canva_design_prompt, CARD_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI()


def create_steps(user_input: str) -> StepBreakdown:
    """
    Break down user input into technical steps for design generation.
    
    Args:
        user_input: User's description of desired content.
    
    Returns:
        StepBreakdown with steps and RAG queries.
    """
    response = use_openai(STEP_PROMPT, user_input, "gpt-4o-mini", format=StepBreakdown)
    return response


def create_canva_functions(user_input: str, page_dimensions: dict) -> str:
    """
    Generate Canva design functions based on user input.
    
    Args:
        user_input: User's description of desired content.
        page_dimensions: Dictionary with 'width' and 'height' in pixels.
    
    Returns:
        JSON string of Canva function definitions.
    """
    all_steps = create_steps(user_input)
    
    # Note: RAG integration is configured separately
    # Using placeholder format for now
    return_type_format = "JSON format matching Canva App SDK methods"
    
    prompt = get_canva_design_prompt(return_type_format, page_dimensions)
    steps = ",".join(all_steps.steps)
    
    response = use_openai(prompt, steps, model="gpt-4o", format=JsonOutput)
    functions = replace_images(response.functions)
    
    return json.dumps(functions, indent=2)


def create_cards_from_user_input(user_input: str, n_cards: Optional[int] = None) -> List[dict]:
    """
    Generate presentation cards based on free-form user input.

    Args:
        user_input: Free-form text from which AI should extract the topic.
        n_cards: Desired number of cards. If None, the AI chooses (3-8).

    Returns:
        List of card dictionaries with 'title' and 'description' keys.
    """
    if n_cards and n_cards >= 1:
        instruction = (
            f'Extract the main topic from the following input:\n\n"{user_input}"\n\n'
            f"Then generate exactly {n_cards} unique presentation cards about that topic."
        )
    else:
        instruction = f'Extract the main topic from the following input:\n\n"{user_input}"\n\n'

    response = use_openai(
        CARD_GENERATION_PROMPT,
        instruction,
        model="gpt-4o-mini",
        format=CardList
    )

    # Convert Pydantic models to dictionaries
    return [
        {"title": card.title, "description": card.description}
        for card in response.cards
    ]