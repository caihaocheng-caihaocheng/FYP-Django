
def S_NER(input_sentence, input_model):
	sentence = input_sentence
	model = input_model
	words_raw = sentence.strip().split(" ")
	preds = model.predict(words_raw)
	return(words_raw, preds)

def return_span_tag(input_str):
    span_sentence = "<span style='color = white;background-color:#f56c6c;'>" + input_str +"</span>"
    return span_sentence


def add_span_tag(words_raw, preds):
    temp1 = []
    temp2 = []
    for i in range(len(preds)):
        if(preds[i] == "O"):
            if(len(temp2) == 0):
                temp1.append(words_raw[i])
            else:
                temp_str =  return_span_tag(" ".join(temp2))
                temp1.append(temp_str)
                temp1.append(words_raw[i])
                temp2.clear()
        else:
            temp2.append(words_raw[i])
            if(i == (len(preds)-1)):
                temp_str = return_span_tag(" ".join(temp2))
                temp1.append(temp_str)
    sentence = " ".join(temp1)
            
    return sentence

