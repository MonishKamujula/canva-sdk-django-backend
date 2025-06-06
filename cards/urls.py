from django.urls import path

from . import views

urlpatterns = [
    path("get_cards", views.get_cards),
    path("create_cards", views.create_cards),
]