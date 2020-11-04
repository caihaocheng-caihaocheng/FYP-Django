preds = ['O' , 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
s = "The China China is playing an increasing role in the United Nation hello"
words_raw = s.strip().split(" ")



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

print(temp1)
print(temp2)
        