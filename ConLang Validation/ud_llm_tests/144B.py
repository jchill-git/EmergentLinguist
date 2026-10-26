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
# the function ignores punctuation
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


def extract_neg_word_position(sentence):
    total_beg, total_prec, total_pre, total_post, total_fol, total_end = 0, 0, 0, 0, 0, 0
    clauses_dict = identify_clauses(sentence)
    for head_id in clauses_dict.keys():
        for dep_id in clauses_dict[head_id]:
            if sentence[dep_id - 1]["upos"] in ["AUX", "PART"] and sentence[dep_id - 1]["feats"] and sentence[dep_id - 1]["feats"].get("Polarity") == "Neg":

                if dep_id == min(clauses_dict[head_id]) and dep_id < len(sentence) and sentence[dep_id - 1]["head"] != sentence[dep_id]["id"]:
                    total_beg += 1
                    break
                elif dep_id != min(clauses_dict[head_id]) and dep_id < len(sentence) and sentence[dep_id - 1]["head"] != sentence[dep_id]["id"] and sentence[dep_id - 1]["head"] > sentence[dep_id - 1]["id"]:
                    total_prec += 1
                    break
                elif dep_id != min(clauses_dict[head_id]) and dep_id < len(sentence) and sentence[dep_id - 1]["head"] == sentence[dep_id]["id"]:
                    total_pre += 1
                    break
                elif dep_id != max(clauses_dict[head_id]) and dep_id > 1 and sentence[dep_id - 1]["head"] == sentence[dep_id - 2]["id"]:
                    total_post += 1
                    break
                elif dep_id != max(clauses_dict[head_id]) and dep_id > 1 and sentence[dep_id - 1]["head"] != sentence[dep_id - 2]["id"] and sentence[dep_id - 1]["head"] < sentence[dep_id - 1]["id"]:
                    total_fol += 1
                    break
                elif dep_id == max(clauses_dict[head_id]) and dep_id > 1 and sentence[dep_id - 1]["head"] != sentence[dep_id - 2]["id"]:
                    total_end += 1
                    break

    return total_beg, total_prec, total_pre, total_post, total_fol, total_end
    

def get_final_answer(orderings):
    if all([x == 0 for x in orderings.values()]):
        return "No separate negative words"

    dominant = max(orderings.items(), key=lambda x: x[1])

    return dominant[0]
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
distrib = {
    "At beginning of clause separated from verb": 0,
    "Preceding verb, separated from verb, but not at beginning of clause": 0,
    "Immediately preverbal": 0,
    "Immediately postverbal": 0,
    "Following verb, separated from verb, but not at end of clause": 0,
    "At end of clause, separated from verb": 0
}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_beg, curr_sent_prec, curr_sent_pre, curr_sent_post, curr_sent_fol, curr_sent_end = extract_neg_word_position(get_sentence_no_mwt(conllu_sent))

    distrib["At beginning of clause separated from verb"] += curr_sent_beg 
    distrib["Preceding verb, separated from verb, but not at beginning of clause"] += curr_sent_prec
    distrib["Immediately preverbal"] += curr_sent_pre
    distrib["Immediately postverbal"] += curr_sent_post
    distrib["Following verb, separated from verb, but not at end of clause"] += curr_sent_fol
    distrib["At end of clause, separated from verb"] += curr_sent_end

# export result as json
result = {
    "distribution": distrib,
    "final_answer": get_final_answer(distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
