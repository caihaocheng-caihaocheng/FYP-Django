from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse

from .model.data_utils import CoNLLDataset
from .model.ner_model import NERModel
from .model.config import Config
from .all_function import *

from .predict_RE import *

from stanfordcorenlp import StanfordCoreNLP

config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

print("DONE!")
# Create your views here.

def index(request):
    return render(request, "NER/base_template.html")

def NER_page(request):
	if(request.method == "POST" and 'submit_NER' in request.POST):
		sentence = request.POST.get("input-sentence")
		print(sentence)
		raw_sen, result = S_NER(sentence,model) # 原句子list， 和 entity list
		whole_sentence = add_span_tag(raw_sen, result)
		entity1, entity2 = return_entity(raw_sen, result)
		front = {"raw_sen": raw_sen, "result": result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence }
		#print(front)
		return render(request, "NER/S_Re.html", front)
	elif(request.method == "POST" and 'submit_stanford' in request.POST):
		nlp = StanfordCoreNLP(r'/Users/caihaocheng/stanford_nlp')
		sentence = request.POST.get("input-sentence")
		stanner = nlp.ner(sentence)
		raw_sen, result = split_stanner(stanner)
		whole_sentence = add_span_tag(raw_sen, result)
		entity1, entity2 = return_stan_entity(raw_sen, result)
		front = {"raw_sen": raw_sen, "result": result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence }
		return render(request, "NER/S_Re.html", front)

	elif(request.method == "POST" and 'submit_RE' in request.POST):
		input_sentence = request.POST.get("input-sentence")
		submit_Entity1 = request.POST.get("submit_Entity1")
		submit_Entity2 = request.POST.get("submit_Entity2")
		#serve as a input for re method 
		submit_Entity_list = [submit_Entity1, submit_Entity2]
		return_Entity1 = '"' + submit_Entity1 + '"'
		return_Entity2 = '"' + submit_Entity2 + '"'
		formatting_sentence = RE_formatting(input_sentence, submit_Entity_list)
		print(formatting_sentence)
		RE_prediction = predict_RE(formatting_sentence)
		output_sentence = RE_add_span(RE_prediction, submit_Entity_list)
		front1 = {"relation": RE_prediction, "output_sentence": output_sentence, 
		"entity1": return_Entity1, "entity2": return_Entity2, "return_sentence": input_sentence }
		return render(request, "NER/S_Re.html", front1)
	else:
		return render(request, "NER/S_Re.html")




def Document_page(request):
    return render(request, "NER/D_RelationExtraction.html")

