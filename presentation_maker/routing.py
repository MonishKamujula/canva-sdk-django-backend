"""
WebSocket URL routing for presentation_maker app.
"""

from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/canva_request", consumers.CanvaRequestConsumer.as_asgi()),
]
