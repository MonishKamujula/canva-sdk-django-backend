"""
Core AI utilities for OpenAI integration.
"""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI()

def use_openai(prompt: str, user_input: str, model: str = "gpt-4o", format=None):
    """
    Send a request to OpenAI with structured output parsing.
    
    Args:
        prompt: System prompt for the AI.
        user_input: User's input message.
        model: OpenAI model to use.
        format: Pydantic model for structured output parsing.
    
    Returns:
        The parsed response object (instance of the format class).
    """
    try:
        completion = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            response_format=format,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise
