"""
Pydantic schemas for the presentation_maker app.

These models define the structure of data used in
presentation generation workflows.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


class StepBreakdown(BaseModel):
    """Breakdown of card content into design steps and RAG queries."""
    steps: List[str]
    rag_query: List[str]


class ListOfDicts(BaseModel):
    """Container for Canva function definitions."""
    functions: List[Dict[str, Any]]


class Card(BaseModel):
    """Single presentation card with title and description."""
    title: str
    description: str


class CardList(BaseModel):
    """Collection of presentation cards."""
    cards: List[Card]
