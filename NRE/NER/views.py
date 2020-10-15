from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse

from .model.data_utils import CoNLLDataset
from .model.ner_model import NERModel
from .model.config import Config
from .all_function import *


config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

print("DONE!")
# Create your views here.

def index(request):
    return render(request, "NER/base_template.html")

def NER_page(request):
	if request.method == "POST":
		sentence = request.POST.get("input-sentence")
		raw_sen, result = S_NER(sentence,model)
		whole_sentence = add_span_tag(raw_sen, result)
		front = {"raw_sen": raw_sen, "result": result, "whole_sentence": whole_sentence}
		return render(request, "NER/S_RelationExtraction.html", front)
	else:
		return render(request, "NER/S_RelationExtraction.html")

def Document_page(request):
    return render(request, "NER/D_RelationExtraction.html")

