"""
Pydantic schemas for the cards app.

These models define the structure of API request/response data
and are used with OpenAI's structured output parsing.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel


class StepBreakdown(BaseModel):
    """Breakdown of user input into technical steps and RAG queries."""
    steps: List[str]
    rag_query: List[str]


class JsonOutput(BaseModel):
    """Container for Canva function definitions."""
    functions: List[Dict]


class Card(BaseModel):
    """Single presentation card with title and description."""
    title: str
    description: str


class CardList(BaseModel):
    """Collection of presentation cards."""
    cards: List[Card]
