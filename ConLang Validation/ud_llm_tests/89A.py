import conllu 
import sys
import json


def get_sentence_no_mwt(sentence):
    new_sent = list()
    for tok in sentence:
        if type(tok["id"]) == int:
            new_sent.append(tok)
    
    return new_sent


def extract_attachment(sentence):
    def check_head(dep_id, head_id):
        left, right, verb = 0, 0, 0
        head_upos = sentence[int(head_id) - 1]["upos"]
        if head_upos == "NOUN":
            if int(dep_id) < int(head_id):
                left += 1
            elif int(dep_id) > int(head_id):
                right += 1

        elif head_upos == "VERB":
            verb += 1
        
        return left, right, verb

    total_left, total_right, total_verb = 0, 0, 0

    for tok in sentence:
        if tok["upos"] == "NUM" and tok["deprel"] == "nummod":
            curr_id = tok["id"]
            curr_head_id = tok["head"]
            curr_left, curr_right, curr_verb = check_head(curr_id, curr_head_id)
            
            total_left += curr_left
            total_right += curr_right
            total_verb += curr_verb
    
    return total_left, total_right, total_verb
    

def get_final_answer(orderings):
    dominant = max(orderings.items(), key=lambda x: x[1])

    return dominant[0]
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
distrib = {"NumN": 0, "NNum": 0, "Numeral only modifies verb": 0}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_left, curr_sent_right, curr_sent_verb = extract_attachment(get_sentence_no_mwt(conllu_sent))

    distrib["NumN"] += curr_sent_left
    distrib["NNum"] += curr_sent_right
    distrib["Numeral only modifies verb"] += curr_sent_verb

# export result as json
result = {
    "distribution": distrib,
    "final_answer": get_final_answer(distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
