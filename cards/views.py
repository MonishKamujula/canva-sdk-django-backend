from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from django.http import JsonResponse
from .models import Cards
from .controllers import create_cards_from_user_input
import random
import json

def get_cards(request):
    cards = Cards.objects.all().values('session_id', 'tittle', 'description')
    return JsonResponse(list(cards), safe=False)

@csrf_exempt
def create_cards(requests):
    data = json.loads(requests.body)
    user_input = data["user_input"]
    session_id = data["session_id"]
    n_cards = int(data["n_cards"])

    print("User input:", user_input, n_cards)

    cards = create_cards_from_user_input(user_input, n_cards)
    print(cards)
    for card in cards:
        Cards.objects.create(
            session_id=session_id,
            tittle=card["title"],
            description=card["description"],
        )

    return JsonResponse(cards, safe=False) #cards
# Create your views here.
