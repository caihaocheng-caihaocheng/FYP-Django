def S_NER(input_sentence, input_model):
	sentence = input_sentence
	model = input_model
	words_raw = sentence.strip().split(" ")
	preds = model.predict(words_raw)
	return words_raw, preds

def return_span_tag(input_str, input_type):
    if input_type == "word1":
        span_sentence = "<span style='color = #fff; background-color:#f56c6c; border-radius: 4px;'>" + input_str +"</span>"
    elif input_type == "word2":
       span_sentence = "<span style='color = #fff; background-color:#67c23a; border-radius: 4px;'>" + input_str +"</span>" 
    else:
       span_sentence = "<span style='color = #fff; background-color: #409eff; border-radius: 4px;'>" + input_str +"</span>" 
    return span_sentence

def add_span_tag(words_raw, preds):
    temp1 = []
    temp2 = []
    for i in range(len(preds)):
        if(preds[i] == "O"):
            if(len(temp2) == 0):
                temp1.append(words_raw[i])
            else:
                temp_str =  return_span_tag(" ".join(temp2), input_type = "word")
                temp1.append(temp_str)
                temp1.append(words_raw[i])
                temp2.clear()
        else:
            temp2.append(words_raw[i])
            if(i == (len(preds)-1)):
                temp_str = return_span_tag(" ".join(temp2), input_type = "word")
                temp1.append(temp_str)
    sentence = " ".join(temp1)
    return sentence

def return_entity(words_raw, preds):
    """
    Maximun for 2 entities 
    input: words_raw, preds. Both are list class
    output entity1, entity2
    """
    # count how many entity is in the sentence 
    count = 0
    for i in range(len(preds)):
        if("B-" in preds[i]):
            count += 1
    #==========================================
    
    if count == 0:
        entity1 = entity2 ='"' + "No entity is suggested!" + '"'
        return entity1, entity2
    elif count == 1:
        temp1 = []
        for i in range(len(preds)):
            if(preds[i] != "O"):
                temp1.append(words_raw[i])
            else:
                pass
        entity1 = " ".join(temp1)
        entity2 = '"' + "No entity is suggested" + '"'
        return entity1, entity2
    elif count==2:
        temp1 = []
        temp2 = []
        temp3 = []
        for i in range(len(preds)):
            #遇到以B- 开头的， 先检查temp3 是否为空，空既是里面的是entity1， 否则将temp3 加入到temp1， 然后清空，再放入temp2
            if("B-" in preds[i]):
                if (len(temp3) != 0 ):
                    for w in range(len(temp3)):
                        temp1.append(temp3[w])
                    temp3.clear()
                    temp3.append(words_raw[i])
                elif(len(temp3) == 0):
                    temp3.append(words_raw[i])
            elif("I-" in preds[i]):
                temp3.append(words_raw[i])
            else:
                pass
        for i in range(len(temp3)):
            temp2.append(temp3[i])
        entity1 = '"' + " ".join(temp1) + '"'
        entity2 = '"' + " ".join(temp2) + '"'
        return entity1, entity2
    else:
        entity1 = entity2 = '"' + "Entity exceeded" + '"'
        return entity1, entity2
    return entity1, entity2

def return_stan_entity(words_raw, preds):
    """
    same as return_entity
    """
    #count how manay entity in the sentence
    store_entity = []
    temp_store = []
    for i in range(len(preds)):
        if(preds[i] != "O"):
            temp_store.append(words_raw[i])
            #print("temp_store: ", temp_store)
        else:
            if(len(temp_store) != 0):
                store_entity.append(temp_store)
                #print("count: ", count)
                temp_store = []
            else:
                pass
    if(len(temp_store) != 0):
        store_entity.append(temp_store)
    #finish counting
    count = len(store_entity)
    
    if count == 0:
        entity1 = entity2 ='"' + "No entity is suggested!" + '"'
    elif count == 1:
        entity1 = " ".join(store_entity[0])
        entity2 ='"' + "No entity is suggested!" + '"'
    elif count == 2:
        entity1 = " ".join(store_entity[0])
        entity2 = " ".join(store_entity[1])
    else:
        entity1 = entity2 ='"' + "Entity exceeded" + '"'
    
    entity1 = '"' + entity1 + '"'
    entity2 = '"' + entity2 + '"'
    return entity1, entity2
        
def RE_formatting(input_sentence, entity_list):
    """
    input: a raw sentence or a sentence list? sentence list seems to be better
    output: a sentence with formatting. e.g. This is e11 entity1 e12 and e21 entity2 e22.
    """
    input_sentence = input_sentence.replace(entity_list[0], "e11 " + entity_list[0] + " e12")
    input_sentence = input_sentence.replace(entity_list[-1], "e21 " + entity_list[-1] + " e22")
    return input_sentence

def RE_add_span(input_relation, entity_list):
    sentence = return_span_tag(entity_list[0], input_type = "word1") + " | " + return_span_tag(entity_list[1], input_type="word1") + " has a " + return_span_tag(input_relation, input_type="relation") + " relation."
    return sentence

def split_stanner(input_result):
    word = []
    entity = []
    for i in range(len(input_result)):
        word.append(input_result[i][0])
        entity.append(input_result[i][1])
    return word, entity

def extract_triple(text):
    # your implementation
    #print('Text: %s.' % text)
    import nltk
    from nltk import pos_tag, word_tokenize, ne_chunk, Tree
    from openie import StanfordOpenIE
    triples = []
    entities = [] #store object and subject
    grammer = "NP: {<JJ>*<NN.*>+}"
    np_chunk = nltk.RegexpParser(grammer)
    tokens = word_tokenize(text)
    result = np_chunk.parse(pos_tag(tokens))
    for i in result:
        if type(i) == Tree:
            word = " ".join(token for token, pos in i.leaves())
            entities.append(word)
    with StanfordOpenIE() as client:
        for sentences in client.annotate(text):
            sentence = []
            for triple in sentences.values():
                sentence.append(triple)
            if (sentence[0] and sentence [2] in entities) and (sentence[1] != "was") and (sentence[1] != "were") and (sentence[1] != "is") and (sentence[1] != "are"):
                triples.append(sentence)
    return triples

def KB(triples):
    # your implementation
    triple = []
    Entities = {}
    Relations = {}
    list1 = []
    list2 =[]
    list3 = []
    #find the unique element
    for entity in triples:
        if entity[0] not in list1:
            list1.append(entity[0])
        if entity[2] not in list1:
            list1.append(entity[2])
            
    #create a dict
    for i in range(len(list1)):
        Entities[i] = list1[i]
    triple.append(Entities)
    #===========================#
    #deal with relation
    for relation in triples:
        if relation[1] not in list2:
            list2.append(relation[1])
    for w in range(len(list2)):
        Relations[w] = list2[w]
    triple.append(Relations)
    #===========================#
    for w in range(len(triples)):
        tem_list = []
        for i in range(len(Entities)):
            if Entities[i] == triples[w][0]:
                tem_list.append(i)
        for q in range(len(Relations)):
            if Relations[q] == triples[w][1]:
                tem_list.append(q)
        for e in range(len(Entities)):
            if Entities[e] == triples[w][2]:
                tem_list.append(e)
        list3.append(tem_list)
    triple.append(list3)
    return triple    

def visualizeKB(kb_input):
    # your implementation
    import graphviz
    dot = graphviz.Digraph('//KB-Demo', filename = 'KB-Demo')
    for key in kb_input[0].keys():
        dot.node( str(key) , label = kb[0][key])
    for triple in kb[2]:
        dot.edge(str(triple[0]), str(triple[2]), label = kb[1][triple[1]])
    return dot

def openie_tool(input_tirple):
    """
    This function is used to extracte entities and relations
    """
    if len(input_tirple) == 1:
        entity1 = input_tirple[0][0]
        entity2 = input_tirple[0][2]
        relation = input_tirple[0][1]
    else:
        entity1 = "Not completed"
        entity2 = "Not completed"
        relation = "Not completed"
    return entity1, entity2, relation

def openie_return_span_sentence(input_e1, input_e2, input_relation):
    entity_list = [input_e1, input_e2]
    sentence = RE_add_span(input_relation, entity_list)
    return sentence

def CNN_predict(input_sentence, model):
    model.load_models("NER/for_cnn/models/")
    result = model.predict(input_sentence)
    word = []
    predict_result = []
    for i in range(len(result)):
        word.append(result[i][0])
        predict_result.append(result[i][1])
    return word, predict_result
    
def BERT_predict(input_sentence, model):
    result = model.predict(input_sentence)
    word, predict_result = [], []
    for i in range(len(result)):
        word.append(result[i]['word'])
        predict_result.append(result[i]['tag'])
    return word, predict_result

def split_para(input_sentence):
    """
    a function to split a paragraph to sentences
    """
    import re
    split_sentence = []
    for element in input_sentence: 
        split_sentence += re.split("(?<=[.!?])\s+", element) 
    return split_sentence

def modify_output():
    return "HELLO"

def KB_to_excel(triples):
    import csv
    # open a file 
    f_entity = open('entity.csv','w',encoding='utf-8')
    print("build")
    csv_entity = csv.writer(f_entity)
    csv_entity.writerow(["entityName","entityId:ID"])
    
    
    f_relation = open('relation.csv','w',encoding='utf-8')
    csv_relation = csv.writer(f_relation)
    csv_relation.writerow([":START_ID",":TYPE",":END_ID"])
    
    triple = []
    Entities = {}
    Relations = {}
    list1 = []
    list2 =[]
    list3 = []
    #find the unique element
    for entity in triples:
        if entity[0] not in list1:
            list1.append(entity[0])
        if entity[2] not in list1:
            list1.append(entity[2])
    #create a dict
    for i in range(len(list1)):
        Entities[i] = list1[i]
    triple.append(Entities)
    
    #write csv 
    for i in Entities:
        csv_entity.writerow([Entities[i],i])
    #===========================#
    
    #deal with relation
    for relation in triples:
        if relation[1] not in list2:
            list2.append(relation[1])
    for w in range(len(list2)):
        Relations[w] = list2[w]
    triple.append(Relations)
    
    #===========================#
    
    for w in range(len(triples)):
        tem_list = []
        for i in range(len(Entities)):
            if Entities[i] == triples[w][0]:
                tem_list.append(i)
        for q in range(len(Relations)):
            if Relations[q] == triples[w][1]:
                tem_list.append(q)
        for e in range(len(Entities)):
            if Entities[e] == triples[w][2]:
                tem_list.append(e)
        list3.append(tem_list)
    triple.append(list3)
    
    for i in range(len(triple[2])):
        csv_relation.writerow([triple[2][i][0],triple[1][triple[2][i][1]],triple[2][i][2]])
    
    
    f_entity.close()
    f_relation.close()
    
    return triple    

def count_entity(pre_list):
    count = 0
    for i in range(len(pre_list)):
        if ("B-" in pre_list[i]):
            count += 1
    return count

def NER_RE_Formatting(sentence_list, model):
    """
    input a list of sentence
    1. do the named entity recognition sentence by sentence
        output: 
    2. based on 1. do relation extraction
    3. format the output [entity1, relation, entitiy2]
    4. output
    """
    triple = []
    for i in range(len(sentence_list)):
        tag = []
        Entity = []
        word, pre_list = BERT_predict(sentence_list[i], model)
        count_entity = count_entity(pre_list)
        if(count_entity == 2):
            entity1, entity2 = return_entity(word, pre_list)
            Entity.append(entity1)
            Entity.append(entity2)
            return 0
        return 0

def create_node(kb_entity_dict):
    """
    input: The dict of entity
    output: a nodes list 
    """
    from pyecharts.charts import Bar
    from pyecharts import options as opts
    from pyecharts import options as opts
    from pyecharts.charts import Graph
    nodes = []
    for i in kb_entity_dict:
        node = opts.GraphNode(name=kb_entity_dict[i], symbol_size=10)
        nodes.append(node)
    return nodes

def create_link(input_kb):
    """
    input: result from kb
    output: a link list
    """
    from pyecharts.charts import Bar
    from pyecharts import options as opts
    from pyecharts import options as opts
    from pyecharts.charts import Graph
    links = []
    kb_entity_dict = input_kb[0]
    rel_dict = input_kb[1]
    kb_rel_dict = input_kb[2]
    for i in range(len(kb_rel_dict)):
        link = opts.GraphLink(source = kb_entity_dict[kb_rel_dict[i][0]], 
                             target = kb_entity_dict[kb_rel_dict[i][2]], 
                             label_opts = rel_dict[kb_rel_dict[i][1]])
        links.append(link)
    return links 








