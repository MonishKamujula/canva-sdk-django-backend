"""
Views for the presentation_maker app.

Handles HTTP endpoints for generating Canva design functions.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .controllers import create_canva_functions

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def canvarequest(request):
    """
    Generate Canva design functions for a presentation card.
    
    Request Body:
        card (dict): Card with 'title' and 'description' keys.
        page_dimensions (dict): Contains 'dimensions' with 'width' and 'height'.
    
    Returns:
        JsonResponse: List of Canva function definitions.
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
        if "card" not in data:
            return JsonResponse(
                {"error": "Missing required field: card"},
                status=400
            )
        
        if "page_dimensions" not in data:
            return JsonResponse(
                {"error": "Missing required field: page_dimensions"},
                status=400
            )
        
        card = data["card"]
        
        # Validate card structure
        if not isinstance(card, dict) or "title" not in card or "description" not in card:
            return JsonResponse(
                {"error": "card must have 'title' and 'description' fields"},
                status=400
            )
        
        # Extract page dimensions
        page_dimensions_wrapper = data["page_dimensions"]
        if "dimensions" not in page_dimensions_wrapper:
            return JsonResponse(
                {"error": "page_dimensions must contain 'dimensions' field"},
                status=400
            )
        
        page_dimensions = page_dimensions_wrapper["dimensions"]
        
        if "width" not in page_dimensions or "height" not in page_dimensions:
            return JsonResponse(
                {"error": "dimensions must have 'width' and 'height' fields"},
                status=400
            )
        
        logger.info(f"Processing canva request for card: {card.get('title', 'Unknown')}")
        
        # Generate Canva functions
        functions_json = create_canva_functions(page_dimensions, card)
        functions = json.loads(functions_json)
        
        logger.info(f"Successfully generated {len(functions)} functions")
        return JsonResponse(functions, safe=False)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response JSON: {e}")
        return JsonResponse(
            {"error": "Failed to generate valid response"},
            status=500
        )
    except Exception as e:
        logger.exception("Error processing canva request")
        return JsonResponse(
            {"error": "Failed to process request", "detail": str(e)},
            status=500
        )