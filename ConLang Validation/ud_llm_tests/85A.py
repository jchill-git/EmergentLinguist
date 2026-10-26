import conllu 
import sys
import json

def extract_ordering(sentence):
    def check_dependents(head_id):
        prepositions, postpositions = 0, 0
        for dep in sentence:
                if dep["head"] == head_id and dep["upos"] == "ADP" and dep["deprel"] == "case":
                    if int(dep["id"]) < int(head_id):
                        prepositions += 1
                    elif int(dep["id"]) > int(head_id):
                        postpositions += 1
        
        return prepositions, postpositions

    total_preps, total_posts = 0, 0

    for tok in sentence:
        if tok["upos"] == "NOUN":
            curr_id = tok["id"]
            curr_preps, curr_posts = check_dependents(curr_id)
            
            total_preps += curr_preps
            total_posts += curr_posts
    
    return total_preps, total_posts
    

def get_final_answer(orderings):
    if all([x == 0 for x in orderings.values()]):
        return "No adpositions"
    
    dominant = max(orderings.items(), key=lambda x: x[1])

    return dominant[0]
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
adpositions_distrib = {"Prepositions": 0, "Postpositions": 0}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_preps, curr_sent_posts = extract_ordering(conllu_sent)

    adpositions_distrib["Prepositions"] += curr_sent_preps
    adpositions_distrib["Postpositions"] += curr_sent_posts

# export result as json
result = {
    "distribution": adpositions_distrib,
    "final_answer": get_final_answer(adpositions_distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
