from django.db import models

class Cards(models.Model):
    session_id = models.CharField(max_length=225)
    tittle = models.CharField(max_length=225)
    description = models.TextField(max_length=450)
# Create your models here.
