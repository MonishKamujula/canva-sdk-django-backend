"""
Shared utilities for external APIs and data processing.
"""

import os
import requests
import logging
from typing import List

logger = logging.getLogger(__name__)

def search_pexels_image(query: str) -> str:
    """
    Search for an image on Pexels by query.
    
    Args:
        query: Search term for the image.
    
    Returns:
        URL of the found image, or a placeholder if not found.
    """
    api_key = os.environ.get("PEXELS_API_KEY", "")
    placeholder_url = "https://i.postimg.cc/jSYRBQWR/image-not-found.png"
    
    if not api_key:
        logger.warning("PEXELS_API_KEY not found in environment variables")
        return placeholder_url
    
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": 1}
    
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        photos = response.json().get('photos', [])
        if photos:
            return photos[0]['src']['medium']
    except requests.RequestException as e:
        logger.warning(f"Pexels API request failed: {e}")
    except (KeyError, IndexError) as e:
        logger.warning(f"Failed to parse Pexels response: {e}")
    
    return placeholder_url


def replace_images(json_data: List[dict]) -> List[dict]:
    """
    Replace image references in JSON data with actual Pexels images.
    
    Args:
        json_data: List of element definitions that may contain 'ref' keys.
    
    Returns:
        Modified list with image URLs replaced.
    """
    for item in json_data:
        if "ref" in item:
            old_ref = item["ref"]
            item["ref"] = search_pexels_image(old_ref)
    return json_data
