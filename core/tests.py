"""
Tests for the core app shared utilities.
"""

from unittest.mock import patch, MagicMock
from django.test import TestCase

from core.utils import search_pexels_image, replace_images
from core.ai import use_openai

class CoreUtilsTest(TestCase):
    """Tests for core utility functions."""
    
    @patch.dict('os.environ', {'PEXELS_API_KEY': ''})
    def test_search_pexels_image_no_key(self):
        """Test missing API key returns placeholder."""
        result = search_pexels_image("test")
        self.assertEqual(result, "https://i.postimg.cc/jSYRBQWR/image-not-found.png")
    
    @patch('core.utils.requests.get')
    @patch.dict('os.environ', {'PEXELS_API_KEY': 'test-key'})
    def test_search_pexels_image_success(self, mock_get):
        """Test successful image search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'photos': [{'src': {'medium': 'https://example.com/image.jpg'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = search_pexels_image("nature")
        self.assertEqual(result, "https://example.com/image.jpg")

    def test_replace_images(self):
        """Test generic replace_images function."""
        with patch('core.utils.search_pexels_image') as mock_search:
            mock_search.return_value = "https://example.com/replaced.jpg"
            data = [{"ref": "query"}, {"other": "value"}]
            result = replace_images(data)
            self.assertEqual(result[0]["ref"], "https://example.com/replaced.jpg")

class CoreAiTest(TestCase):
    """Tests for OpenAI wrapper."""
    
    @patch('core.ai.openai_client.beta.chat.completions.parse')
    def test_use_openai_success(self, mock_parse):
        """Test successful OpenAI call."""
        mock_parsed_obj = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(parsed=mock_parsed_obj))
        ]
        mock_parse.return_value = mock_completion
        
        result = use_openai("prompt", "input", format=MagicMock())
        self.assertEqual(result, mock_parsed_obj)
        
        # Verify correct args usage
        call_kwargs = mock_parse.call_args[1]
        self.assertIn("messages", call_kwargs)
        self.assertIn("response_format", call_kwargs)
