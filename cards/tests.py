"""
Tests for the cards app.

Includes unit tests for controllers and integration tests for API endpoints.
"""

import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
from django.urls import reverse

from .models import Cards
from .controllers import create_cards_from_user_input
from core.utils import search_pexels_image, replace_images


class CardsModelTest(TestCase):
    """Tests for the Cards model."""
    
    def test_create_card(self):
        """Test creating a card."""
        card = Cards.objects.create(
            session_id="test-session-123",
            title="Test Title",
            description="Test description"
        )
        self.assertEqual(card.session_id, "test-session-123")
        self.assertEqual(card.title, "Test Title")
        self.assertEqual(card.description, "Test description")
    
    def test_card_str(self):
        """Test card string representation."""
        card = Cards.objects.create(
            session_id="test-session",
            title="My Card",
            description="Description"
        )
        # Cards model doesn't have __str__ yet, but should work
        self.assertIsNotNone(str(card))


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
        mock_get.assert_called_once()
    
    @patch('core.utils.requests.get')
    @patch.dict('os.environ', {'PEXELS_API_KEY': 'test-key'})
    def test_empty_results_returns_placeholder(self, mock_get):
        """Test empty search results return placeholder."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'photos': []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = search_pexels_image("xyz123nonexistent")
        
        self.assertEqual(result, "https://i.postimg.cc/jSYRBQWR/image-not-found.png")


class ReplaceImagesTest(TestCase):
    """Tests for the replace_images function."""
    
    @patch('core.utils.search_pexels_image')
    def test_replaces_ref_fields(self, mock_search):
        """Test that 'ref' fields are replaced with image URLs."""
        mock_search.return_value = "https://example.com/replaced.jpg"
        
        data = [
            {"type": "image", "ref": "sunset"},
            {"type": "text", "content": "Hello"},
            {"type": "image", "ref": "ocean"},
        ]
        
        result = replace_images(data)
        
        self.assertEqual(result[0]["ref"], "https://example.com/replaced.jpg")
        self.assertEqual(result[1]["content"], "Hello")
        self.assertEqual(result[2]["ref"], "https://example.com/replaced.jpg")
        self.assertEqual(mock_search.call_count, 2)


class CardsViewTest(TestCase):
    """Integration tests for cards API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = Client()
    
    def test_get_cards_empty(self):
        """Test GET cards returns empty list when no cards exist."""
        response = self.client.get('/cards/get_cards')
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])
    
    def test_get_cards_with_data(self):
        """Test GET cards returns list of cards."""
        Cards.objects.create(
            session_id="session-1",
            title="Card 1",
            description="Description 1"
        )
        Cards.objects.create(
            session_id="session-2",
            title="Card 2",
            description="Description 2"
        )
        
        response = self.client.get('/cards/get_cards')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
    
    def test_create_cards_invalid_json(self):
        """Test POST with invalid JSON returns 400."""
        response = self.client.post(
            '/cards/create_cards',
            data="not valid json",
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())
    
    def test_create_cards_missing_fields(self):
        """Test POST with missing fields returns 400."""
        response = self.client.post(
            '/cards/create_cards',
            data=json.dumps({"user_input": "test"}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing required fields', response.json()['error'])
    
    def test_create_cards_invalid_n_cards(self):
        """Test POST with invalid n_cards returns 400."""
        response = self.client.post(
            '/cards/create_cards',
            data=json.dumps({
                "user_input": "test",
                "session_id": "session-1",
                "n_cards": "not a number"
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('n_cards', response.json()['error'])
    
    def test_get_cards_wrong_method(self):
        """Test POST to get_cards returns 405."""
        response = self.client.post('/cards/get_cards')
        
        self.assertEqual(response.status_code, 405)
    
    def test_create_cards_wrong_method(self):
        """Test GET to create_cards returns 405."""
        response = self.client.get('/cards/create_cards')
        
        self.assertEqual(response.status_code, 405)
    
    @patch('cards.views.create_cards_from_user_input')
    def test_create_cards_success(self, mock_create):
        """Test successful card creation."""
        mock_create.return_value = [
            {"title": "Card 1", "description": "Desc 1"},
            {"title": "Card 2", "description": "Desc 2"},
        ]
        
        response = self.client.post(
            '/cards/create_cards',
            data=json.dumps({
                "user_input": "Create cards about Python",
                "session_id": "test-session",
                "n_cards": 2
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        
        # Verify cards were saved to database
        saved_cards = Cards.objects.filter(session_id="test-session")
        self.assertEqual(saved_cards.count(), 2)
