from django.contrib import admin

# Register your models here.

from .models import Cards

# admin.site.register(Cards)
class CardsAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'tittle', 'description')

admin.site.register(Cards, CardsAdmin)

