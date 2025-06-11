from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .controllers import create_canva_functions, create_steps
import json

# Create your views here.
@csrf_exempt
def canvarequest(requests):
    data = json.loads(requests.body)
    card = data['card']
    page_dimensions = data['page_dimensions']["dimensions"]

    functions = json.loads(create_canva_functions(page_dimensions, card))
    
    return JsonResponse(functions, safe=False)