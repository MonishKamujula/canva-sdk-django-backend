"""
WebSocket consumers for the presentation_maker app.

Handles real-time streaming of Canva design elements.
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

from .controllers import stream_canva_functions

logger = logging.getLogger(__name__)


class CanvaRequestConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming Canva design elements.
    
    Protocol:
        1. Client connects to ws://host/ws/canva_request
        2. Client sends JSON: {"card": {...}, "page_dimensions": {...}}
        3. Server streams back individual elements as they are generated
        4. Server sends {"type": "complete"} when done
        5. Server closes connection
    """

    async def connect(self):
        """Accept the WebSocket connection."""
        await self.accept()
        logger.info("WebSocket connection accepted")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        """
        Handle incoming WebSocket message.
        
        Expected payload:
        {
            "card": {"title": "...", "description": "..."},
            "page_dimensions": {"dimensions": {"width": ..., "height": ...}}
        }
        """
        try:
            data = json.loads(text_data)
            logger.info(f"Received WebSocket message: {data.get('card', {}).get('title', 'Unknown')}")
            
            # Validate required fields
            if "card" not in data:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "Missing required field: card"
                }))
                return
            
            if "page_dimensions" not in data:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "Missing required field: page_dimensions"
                }))
                return
            
            card = data["card"]
            page_dimensions_wrapper = data["page_dimensions"]
            
            # Validate card structure
            if not isinstance(card, dict) or "title" not in card or "description" not in card:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "card must have 'title' and 'description' fields"
                }))
                return
            
            # Extract page dimensions
            if "dimensions" not in page_dimensions_wrapper:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "page_dimensions must contain 'dimensions' field"
                }))
                return
            
            page_dimensions = page_dimensions_wrapper["dimensions"]
            
            if "width" not in page_dimensions or "height" not in page_dimensions:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": "dimensions must have 'width' and 'height' fields"
                }))
                return
            
            # Stream the Canva functions
            element_count = 0
            async for element in stream_canva_functions(page_dimensions, card):
                element_count += 1
                await self.send(text_data=json.dumps({
                    "type": "element",
                    "index": element_count,
                    "data": element
                }))
                logger.debug(f"Sent element {element_count}")
            
            # Send completion message
            await self.send(text_data=json.dumps({
                "type": "complete",
                "total_elements": element_count
            }))
            logger.info(f"Streaming complete. Sent {element_count} elements.")
            
            # Close the connection
            await self.close()
            
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "Invalid JSON in request"
            }))
        except Exception as e:
            logger.exception("Error processing WebSocket message")
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": f"Failed to process request: {str(e)}"
            }))
