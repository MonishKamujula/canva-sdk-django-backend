from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from django.http import JsonResponse
from .models import Cards
from .controllers import create_cards_from_user_input
import random

def get_cards(request):
    cards = Cards.objects.all().values('session_id', 'tittle', 'description')
    return JsonResponse(list(cards), safe=False)

@csrf_exempt
def create_cards(requests):
    user_input = requests.POST.get("user_input")
    n_cards = requests.POST.get("n_cards")

    cards = create_cards_from_user_input(user_input, n_cards)
    print(cards)
    for card in cards:
        Cards.objects.create(
            session_id=random.randint(1, 1000),
            tittle=card["title"],
            description=card["description"],
        )

    return cards
# Create your views here.
