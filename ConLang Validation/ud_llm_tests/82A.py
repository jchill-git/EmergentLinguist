import conllu 
from collections import Counter
import sys
import json


def extract_constituent_ordering(sentence):
    def check_dependents(head_id):
        ordering = {"S": None, "V": head_id}
        for dep in sentence:
                if dep["head"] == head_id:
                    if dep["deprel"] in ["nsubj"] and dep["upos"] in ["NOUN"]:
                        ordering["S"] = dep["id"]
        
        if any([x is None for x in ordering.values()]):
            return None
        else:
            return "".join(sorted(ordering.keys(), key=lambda x: ordering[x]))

    orderings = list()

    for tok in sentence:
        if tok["upos"] == "VERB" or tok["deprel"] == "root":
            curr_id = tok["id"]
            curr_ordering = check_dependents(curr_id)
            
            if curr_ordering:
                orderings.append(curr_ordering)
    
    if len(orderings) > 0:
        return orderings
    else:
        return None
    

def get_dominant_ordering(orderings):
    if len(orderings) > 0:
        dominant = max(orderings.items(), key=lambda x: x[1])
        return dominant[0]

    else:
        return "No valid orderings"
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
ordering_counter = Counter()

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_ordering_list = extract_constituent_ordering(conllu_sent)

    if curr_ordering_list:
        for curr_ordering in curr_ordering_list:
            ordering_counter[curr_ordering] += 1

# export result as json
result = {
    "distribution": ordering_counter,
    "final_answer": get_dominant_ordering(ordering_counter)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
