import conllu 
import sys
import json
from collections import defaultdict


def get_sentence_no_mwt(sentence):
    new_sent = list()
    for tok in sentence:
        if type(tok["id"]) == int:
            new_sent.append(tok)
    
    return new_sent


# a function that accepts a conllu sentence and returns a dictionary of all the clause head ids and the ids of their dependents
# the function ignores punctuation and multi-word tokens
def identify_clauses(sentence):
    clause_boundaries = ["root", "conj", "parataxis", "csubj", "ccomp", "xcomp", "advcl", "acl"]

    result = defaultdict(list)
    for tok in sentence:
        if tok["upos"] == "PUNCT" or tok["id"] == tuple:
            continue

        current_deprel = tok["deprel"]
        current_id = tok["id"]

        if current_deprel in clause_boundaries:
            result[tok["id"]].append(tok["id"])
            continue

        while current_deprel not in clause_boundaries:
            current_id = sentence[current_id - 1]["head"]
            current_deprel = sentence[current_id - 1]["deprel"]

        result[current_id].append(tok["id"])

    for k, v in result.items():
        result[k] = sorted(v)
    
    return result


def extract_subordinator_position(sentence):
    total_beg, total_end, total_int = 0, 0, 0
    clauses_dict = identify_clauses(sentence)
    for head_id in clauses_dict.keys():
        for dep_id in clauses_dict[head_id]:
            if sentence[dep_id - 1]["deprel"] == "mark" and sentence[head_id - 1]["deprel"] in ["advcl", "advcl:relcl"]:

                if dep_id == min(clauses_dict[head_id]):
                    total_beg += 1
                elif dep_id == max(clauses_dict[head_id]):
                    total_end += 1
                else:
                    total_int += 1
                break

    return total_beg, total_end, total_int
    

def get_final_answer(orderings):
    if len(orderings) > 0:
        dominant = max(orderings.items(), key=lambda x: x[1])
        return dominant[0]

    else:
        return "No valid orderings"
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
distrib = {
    "Adverbial subordinators which appear at the beginning of the subordinate clause": 0,
    "Adverbial subordinators which appear at the end of the subordinate clause": 0,
    "Clause-internal adverbial subordinators": 0
}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_beg, curr_sent_end, curr_sent_int = extract_subordinator_position(get_sentence_no_mwt(conllu_sent))

    distrib["Adverbial subordinators which appear at the beginning of the subordinate clause"] += curr_sent_beg
    distrib["Adverbial subordinators which appear at the end of the subordinate clause"] += curr_sent_end
    distrib["Clause-internal adverbial subordinators"] += curr_sent_int

# export result as json
result = {
    "distribution": distrib,
    "final_answer": get_final_answer(distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
