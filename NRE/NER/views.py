from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse
import sys
import torch

sys.path.append('../')
from .rel_extract.extractor import RelationExtractor
from .rel_extract.model import RelTaggerModel
from transformers import BertModel
from transformers import BertTokenizer
import os
import pandas as pd
import json

from .model.data_utils import CoNLLDataset
from .model.ner_model import NERModel
from .model.config import Config
from .all_function import *
from .predict_RE import *
from random import randrange
from .for_bert.bert import Ner
from .for_cnn.ner import Parser

from stanfordcorenlp import StanfordCoreNLP
import nltk
from nltk import pos_tag, word_tokenize, ne_chunk, Tree
from openie import StanfordOpenIE
from graphviz import Digraph
from openie import StanfordOpenIE
import json
from rest_framework.views import APIView

from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts import options as opts
from pyecharts.charts import Graph

from pyjarowinkler import distance
import json,math, nltk
from nltk.corpus import brown
from pprint import pprint

import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np

config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)
print("Successfully import CRF model")

BERT_model = Ner("NER/for_bert/out_base/")
print("Successfully import BERT model")

CNN_model = Parser()
# Create your views here.
def response_as_json(data):
    json_str = json.dumps(data)
    response = HttpResponse(
        json_str,
        content_type="application/json",
    )
    response["Access-Control-Allow-Origin"] = "*"
    return response

def json_response(data, code=200):
    data = {
        "code": code,
        "msg": "success",
        "data": data,
    }
    return response_as_json(data)

def json_error(error_string="error", code=500, **kwargs):
    data = {
        "code": code,
        "msg": error_string,
        "data": {}
    }
    data.update(kwargs)
    return response_as_json(data)

JsonResponse = json_response
JsonError = json_error

#======================================================================================================
_filename = 'NER/save_bert_large_with_squad0'
#UNCASED = './bert-large-uncased-whole-word-masking-finetuned-squad'
VOCAB = 'vocab.txt'
pretrained_weights = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model_class = BertModel
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
UNCASED = 'NER/lyh_bert_model'
language_model = BertModel.from_pretrained(UNCASED)
print("Successfully load language model")
language_model.resize_token_embeddings(len(tokenizer))
re_model = RelTaggerModel(language_model)
checkpoint = torch.load(_filename)
re_model.load_state_dict(checkpoint['model_state_dict'], True)
print("Successfully load checkpoint")
relations = ['editor',
 'elector',
 'debut participant',
 'maintained by',
 'commissioned by',
 'cast member',
 'noble title',
 'studied by',
 'studies',
 'admissible rule in',
 'author',
 'family',
 'has contributing factor',
 'named by',
 'cause of death',
 'owner of',
 'solves',
 'location',
 'chief executive officer',
 'lifestyle',
 'promoted',
 'time period',
 'home port',
 'record held',
 'applies to jurisdiction',
 'stated in',
 'nominee',
 'route of administration',
 'mother',
 'country of citizenship',
 'spouse',
 'sex or gender',
 'place of death',
 'father',
 'field of this occupation',
 'appears in the form of',
 'interested in',
 'partnership with',
 'professorship',
 'manufacturer',
 'parliamentary term',
 'solved by',
 'represented by',
 'has part'
 'significant event',
 'commanded by',
 'hair color',
 'occupation of a person',
 'noble title',
 'elector',
 'capital of',
 'candidate',
 'employer',
 'type of kinship',
 'has cause',     
 'country of origin',
 'approved by',
 'significant person',
 'relative',
 'work location',
 'named after',
 'connects with',
 'partner',]
extractor = RelationExtractor(re_model, tokenizer, relations)
#=========================================================================================================



def index(request):
    return render(request, "NER/base_template.html")

def NER_page(request):
	if(request.method == "POST" and 'submit_NER' in request.POST):
		sentence = request.POST.get("input-sentence")
		raw_sen, result = S_NER(sentence, model) # 原句子list， 和 entity list
		whole_sentence = add_span_tag(raw_sen, result)
		entity1, entity2 = return_entity(raw_sen, result)
		front = {"raw_sen": raw_sen, "result": result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence }
		#print(front)
		return render(request, "NER/S_Re.html", front)

	elif(request.method == "POST" and 'submit_stanford' in request.POST):
		nlp = StanfordCoreNLP(r'/Users/caihaocheng/stanfordnlp_resources/stanford-corenlp-full-2018-10-05')
		sentence = request.POST.get("input-sentence")
		stanner = nlp.ner(sentence)
		raw_sen, result = split_stanner(stanner)
		whole_sentence = add_span_tag(raw_sen, result)
		entity1, entity2 = return_stan_entity(raw_sen, result)
		front = {"raw_sen": raw_sen, "result": result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence }
		return render(request, "NER/S_Re.html", front)

	elif(request.method == "POST" and 'submit_CNN' in request.POST):
		sentence = request.POST.get("input-sentence")
		word, predict_result = CNN_predict(sentence, CNN_model)
		whole_sentence = add_span_tag(word, predict_result)
		entity1, entity2 = return_entity(word, predict_result)
		front = {"raw_sen": word, "result": predict_result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence}
		return render(request, "NER/S_RE.html", front)

	elif(request.method == "POST" and 'submit_BERT' in request.POST):
		sentence = request.POST.get("input-sentence")
		word, predict_result = BERT_predict(sentence, BERT_model)
		whole_sentence = add_span_tag(word, predict_result)
		entity1, entity2 = return_entity(word, predict_result)
		front = {"raw_sen": word, "result": predict_result, "whole_sentence": whole_sentence,
		 "entity1": entity1, "entity2": entity2, "return_sentence": sentence}
		return render(request, "NER/S_RE.html", front)

	elif(request.method == "POST" and 'submit_RE' in request.POST):
		input_sentence = request.POST.get("input-sentence")
		submit_Entity1 = request.POST.get("submit_Entity1")
		submit_Entity2 = request.POST.get("submit_Entity2")
		#serve as a input for re method 
		#w1 = extractor.rank(text='John Smith received an OBE', head='John Smith', tail='OBE')
		#print(w1)
		submit_Entity_list = [submit_Entity1, submit_Entity2]
		return_Entity1 = '"' + submit_Entity1 + '"'
		return_Entity2 = '"' + submit_Entity2 + '"'
		re_result = extractor.rank(text=input_sentence, head=submit_Entity1, tail=submit_Entity2)[0][0]
		#formatting_sentence = RE_formatting(input_sentence, submit_Entity_list)
		#print(formatting_sentence)
		#RE_prediction = predict_RE(formatting_sentence)
		output_sentence = RE_add_span(re_result, submit_Entity_list)
		front1 = {"output_sentence":output_sentence,"relation": re_result, "entity1": return_Entity1, "entity2": return_Entity2, "return_sentence": input_sentence }
		return render(request, "NER/S_Re.html", front1)
	elif(request.method == "POST" and 'submit_openie' in request.POST):
		input_sentence = request.POST.get("input-sentence")
		triple = extract_triple(input_sentence)
		return_Entity1, return_Entity2, return_Relation = openie_tool(triple)
		output_sentence = openie_return_span_sentence(return_Entity1, return_Entity2, return_Relation)
		front = {"relation": return_Relation, "output_sentence": output_sentence, 
		"entity1": return_Entity1, "entity2": return_Entity2, "return_sentence": input_sentence }
		return render(request, "NER/S_Re.html", front)
	else:
		return render(request, "NER/S_Re.html")

temp = (
	Graph()
	.dump_options_with_quotes()
)
def Document_page(request):
	if(request.method=="POST" and 'openie-submit' in request.POST):
		input_para = request.POST.get("input-paragraph")
		sentence_list = split_para(input_para)
		#Named Entity Recognition
		Total_result = BERT_split_para_predict(sentence_list, BERT_model)
		format_result = format_bert_ner(Total_result)
		#Relation Extraction
		s_out = sentence_list_re(format_result, extractor)
		tr = KB(s_out)
		#print(tr)
		f_result1 = get_clustered_triples(get_triple_point(get_triple(tr)), get_triple_sim(get_triple(tr),tr),tr)

		KB_result_test = KB(f_result1)
		nodes = create_node(KB_result_test[0])
		links = create_link(KB_result_test)
		global c
		c = (
			Graph()
			.add("", 
			nodes, 
			links, 
			repulsion=8000, 
			edge_symbol= ['arrow'], edge_label=opts.LabelOpts(
        is_show=True, position="middle", formatter="{c}"),
		is_draggable = True,
        )
			.set_global_opts(title_opts=opts.TitleOpts(title="Relation Visulization"))
			.dump_options_with_quotes()

			)
		class ChartView(APIView):
			def get(self, request, *args, **kwargs):
				return JsonResponse(json.loads(c))	
		front = {"return_sentence": input_para, "nodes": nodes, "links": links}
		return render(request, "NER/D_RelationExtraction.html", front)	
	elif(request.method=="POST" and 'BERT-NER' in request.POST):
		input_para = request.POST.get("input-paragraph")
		return "HELLO"
	else:
		c = (
			Graph()
			.dump_options_with_quotes()
		)	
		return render(request, "NER/D_RelationExtraction.html")

class ChartView(APIView):
    def get(self, request, *args, **kwargs):
        return JsonResponse(json.loads(c))