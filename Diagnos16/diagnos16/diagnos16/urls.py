from django.contrib import admin
from django.urls import path

from webapp.views import home, doctores, diagnostic
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('home', home),
    path('doctores', doctores),
    path('diagnostic', diagnostic),
]
