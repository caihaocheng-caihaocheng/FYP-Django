
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
        entity1 = store_entity[0]
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
    sentence = return_span_tag(entity_list[0], input_type = "word1") + " | " + return_span_tag(entity_list[1], input_type="word2") + " has a " + return_span_tag(input_relation, input_type="relation") + " relation."
    return sentence

def split_stanner(input_result):
    word = []
    entity = []
    for i in range(len(input_result)):
        word.append(input_result[i][0])
        entity.append(input_result[i][1])
    return word, entity