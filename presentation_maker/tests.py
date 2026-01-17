"""
Tests for the presentation_maker app.

Includes unit tests for controllers and integration tests for API endpoints.
"""

import json
import math
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client

from .controllers import estimate_pixels
from core.utils import search_pexels_image, replace_images


class SearchPexelsImageTest(TestCase):
    """Tests for the search_pexels_image function."""
    
    @patch.dict('os.environ', {'PEXELS_API_KEY': ''})
    def test_missing_api_key_returns_placeholder(self):
        """Test that missing API key returns placeholder image."""
        result = search_pexels_image("test query")
        self.assertEqual(result, "https://i.postimg.cc/jSYRBQWR/image-not-found.png")
    
    @patch('core.utils.requests.get')
    @patch.dict('os.environ', {'PEXELS_API_KEY': 'test-key'})
    def test_successful_image_search(self, mock_get):
        """Test successful image search returns URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'photos': [{'src': {'medium': 'https://example.com/image.jpg'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = search_pexels_image("nature")
        
        self.assertEqual(result, "https://example.com/image.jpg")
    
    @patch('core.utils.requests.get')
    @patch.dict('os.environ', {'PEXELS_API_KEY': 'test-key'})
    def test_api_error_returns_placeholder(self, mock_get):
        """Test API error returns placeholder."""
        import requests as req
        mock_get.side_effect = req.RequestException("API Error")
        
        result = search_pexels_image("test")
        
        self.assertEqual(result, "https://i.postimg.cc/jSYRBQWR/image-not-found.png")


class ReplaceImagesTest(TestCase):
    """Tests for the replace_images function."""
    
    @patch('core.utils.search_pexels_image')
    def test_replaces_ref_fields(self, mock_search):
        """Test that 'ref' fields are replaced with image URLs."""
        mock_search.return_value = "https://example.com/image.jpg"
        
        data = [
            {"type": "image", "ref": "sunset"},
            {"type": "text", "content": "Hello"},
        ]
        
        result = replace_images(data)
        
        self.assertEqual(result[0]["ref"], "https://example.com/image.jpg")
        self.assertEqual(result[1]["content"], "Hello")
        mock_search.assert_called_once_with("sunset")
    
    def test_empty_list(self):
        """Test with empty list."""
        result = replace_images([])
        self.assertEqual(result, [])


class CreateCanvaFunctionsTest(TestCase):
    """Tests for create_canva_functions controller."""
    
    @patch('presentation_maker.controllers.create_steps')
    @patch('presentation_maker.controllers.handle_rag')
    @patch('presentation_maker.controllers.create_react_agent')
    @patch('presentation_maker.controllers.ChatOpenAI')
    @patch('presentation_maker.controllers.replace_images')
    def test_strips_markdown_from_response(self, mock_replace, mock_chat, mock_agent, mock_rag, mock_steps):
        """Test that markdown code blocks are stripped from agent response."""
        from presentation_maker.controllers import create_canva_functions
        
        # Setup mocks
        mock_steps.return_value.rag_query = ["query"]
        mock_steps.return_value.steps = ["step1"]
        mock_rag.return_value = "docs"
        
        # Mock agent response with markdown
        mock_agent_executor = MagicMock()
        mock_agent_executor.invoke.return_value = {
            "messages": [
                MagicMock(content='```json\n[{"type": "text"}]\n```')
            ]
        }
        mock_agent.return_value = mock_agent_executor
        
        mock_replace.side_effect = lambda x: x # Identity function
        
        # Execute
        result = create_canva_functions(
            {"width": 800, "height": 600}, 
            {"title": "Valid Title", "description": "Valid Description"}
        )
        
        # Verify
        self.assertEqual(json.loads(result), [{"type": "text"}])


class EstimatePixelsTest(TestCase):
    """Tests for the estimate_pixels tool function."""
    
    def test_single_line_text(self):
        """Test height estimation for single line text."""
        # Short text that fits in one line
        result = estimate_pixels.invoke({
            "content": "Hello",
            "box_width_px": 500,
            "font_size_pt": 16,
            "mu": 0.56,
            "debug": False
        })
        
        # Should return a positive float
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_multi_line_text(self):
        """Test height estimation for multi-line text."""
        # Long text that wraps to multiple lines
        long_text = "This is a very long text that should definitely wrap to multiple lines when displayed in a narrow text box."
        
        result = estimate_pixels.invoke({
            "content": long_text,
            "box_width_px": 200,
            "font_size_pt": 16,
            "mu": 0.56,
            "debug": False
        })
        
        # Multi-line should be taller than single line
        single_line_result = estimate_pixels.invoke({
            "content": "Short",
            "box_width_px": 200,
            "font_size_pt": 16,
            "mu": 0.56,
            "debug": False
        })
        
        self.assertGreater(result, single_line_result)
    
    def test_larger_font_increases_height(self):
        """Test that larger font increases height."""
        small_font_result = estimate_pixels.invoke({
            "content": "Test text",
            "box_width_px": 300,
            "font_size_pt": 12,
            "mu": 0.56,
            "debug": False
        })
        
        large_font_result = estimate_pixels.invoke({
            "content": "Test text",
            "box_width_px": 300,
            "font_size_pt": 24,
            "mu": 0.56,
            "debug": False
        })
        
        self.assertGreater(large_font_result, small_font_result)


class CanvaRequestViewTest(TestCase):
    """Integration tests for canva request API endpoint."""
    
    def setUp(self):
        """Set up test client."""
        self.client = Client()
    
    def test_invalid_json(self):
        """Test POST with invalid JSON returns 400."""
        response = self.client.post(
            '/presentation_maker/canva_request',
            data="not valid json",
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())
    
    def test_missing_card_field(self):
        """Test POST without card field returns 400."""
        response = self.client.post(
            '/presentation_maker/canva_request',
            data=json.dumps({
                "page_dimensions": {"dimensions": {"width": 800, "height": 600}}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('card', response.json()['error'])
    
    def test_missing_page_dimensions_field(self):
        """Test POST without page_dimensions returns 400."""
        response = self.client.post(
            '/presentation_maker/canva_request',
            data=json.dumps({
                "card": {"title": "Test", "description": "Test description"}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('page_dimensions', response.json()['error'])
    
    def test_invalid_card_structure(self):
        """Test POST with invalid card structure returns 400."""
        response = self.client.post(
            '/presentation_maker/canva_request',
            data=json.dumps({
                "card": {"only_title": "Missing description"},
                "page_dimensions": {"dimensions": {"width": 800, "height": 600}}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('title', response.json()['error'])
    
    def test_missing_dimensions_nested_field(self):
        """Test POST with missing nested dimensions returns 400."""
        response = self.client.post(
            '/presentation_maker/canva_request',
            data=json.dumps({
                "card": {"title": "Test", "description": "Test description"},
                "page_dimensions": {}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('dimensions', response.json()['error'])
    
    def test_wrong_http_method(self):
        """Test GET to canva_request returns 405."""
        response = self.client.get('/presentation_maker/canva_request')
        
        self.assertEqual(response.status_code, 405)
    
    @patch('presentation_maker.views.create_canva_functions')
    def test_successful_request(self, mock_create):
        """Test successful canva request."""
        mock_create.return_value = json.dumps([
            {"type": "text", "content": "Hello World"},
            {"type": "image", "ref": "https://example.com/image.jpg"}
        ])
        
        response = self.client.post(
            '/presentation_maker/canva_request',
            data=json.dumps({
                "card": {
                    "title": "Test Presentation",
                    "description": "Topics: intro, conclusion"
                },
                "page_dimensions": {
                    "dimensions": {"width": 1920, "height": 1080}
                }
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
