"""
Views for the cards app.

Handles HTTP endpoints for creating and retrieving presentation cards.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import Cards
from .controllers import create_cards_from_user_input

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def get_cards(request):
    """
    Retrieve all cards from the database.
    
    Returns:
        JsonResponse: List of cards with session_id, title, and description.
    """
    try:
        cards = Cards.objects.all().values('session_id', 'title', 'description')
        return JsonResponse(list(cards), safe=False)
    except Exception as e:
        logger.exception("Error retrieving cards")
        return JsonResponse(
            {"error": "Failed to retrieve cards", "detail": str(e)},
            status=500
        )


@csrf_exempt
@require_http_methods(["POST"])
def create_cards(request):
    """
    Create new presentation cards based on user input.
    
    Request Body:
        user_input (str): Free-form text describing the presentation topic.
        session_id (str): Unique session identifier.
        n_cards (int): Number of cards to generate.
    
    Returns:
        JsonResponse: List of generated cards.
    """
    try:
        # Parse and validate request body
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse(
                {"error": "Invalid JSON in request body"},
                status=400
            )
        
        # Validate required fields
        required_fields = ["user_input", "session_id", "n_cards"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return JsonResponse(
                {"error": f"Missing required fields: {', '.join(missing_fields)}"},
                status=400
            )
        
        user_input = data["user_input"]
        session_id = data["session_id"]
        
        try:
            n_cards = int(data["n_cards"])
            if n_cards < 1:
                raise ValueError("n_cards must be positive")
        except (ValueError, TypeError) as e:
            return JsonResponse(
                {"error": "n_cards must be a positive integer"},
                status=400
            )
        
        logger.info(f"Creating {n_cards} cards for session {session_id}")
        
        # Generate cards using AI
        cards = create_cards_from_user_input(user_input, n_cards)
        
        # Save cards to database
        for card in cards:
            Cards.objects.create(
                session_id=session_id,
                title=card["title"],
                description=card["description"],
            )
        
        logger.info(f"Successfully created {len(cards)} cards")
        return JsonResponse(cards, safe=False)
        
    except Exception as e:
        logger.exception("Error creating cards")
        return JsonResponse(
            {"error": "Failed to create cards", "detail": str(e)},
            status=500
        )
