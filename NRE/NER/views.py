from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    return render(request, "NER/base_template.html")

def NER_page(request):
    i = "hello"
    return render(request, "NER/S_RelationExtraction.html", {"test": i})