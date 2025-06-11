from django.urls import path

from . import views

urlpatterns = [
    path("canva_request", views.canvarequest),
]