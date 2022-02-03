from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from pygments import console


def home(request):
    return render(request, 'webapp/index.html')


def doctores(request):
    return render(request, 'webapp/doctores.html')


def diagnostic(request):

    return render(request, 'webapp/diagnostic.html')