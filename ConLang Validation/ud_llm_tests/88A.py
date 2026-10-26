import conllu 
import sys
import json

def extract_ordering(sentence):
    def check_dependents(head_id):
        left, right = 0, 0
        for dep in sentence:
                if dep["head"] == head_id and dep["feats"] and dep["feats"].get("PronType") == "Dem" and dep["deprel"] == "det":
                    if int(dep["id"]) < int(head_id):
                        left += 1
                    elif int(dep["id"]) > int(head_id):
                        right += 1
        
        return left, right

    total_left, total_right = 0, 0

    for tok in sentence:
        if tok["upos"] == "NOUN":
            curr_id = tok["id"]
            curr_left, curr_right = check_dependents(curr_id)
            
            total_left += curr_left
            total_right += curr_right
    
    return total_left, total_right
    

def get_final_answer(orderings):
    if all([x == 0 for x in orderings.values()]):
        return "No demonstratives"
    
    dominant = max(orderings.items(), key=lambda x: x[1])

    return dominant[0]
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
distrib = {"DemN": 0, "NDem": 0}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_left, curr_sent_right = extract_ordering(conllu_sent)

    distrib["DemN"] += curr_sent_left
    distrib["NDem"] += curr_sent_right

# export result as json
result = {
    "distribution": distrib,
    "final_answer": get_final_answer(distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
