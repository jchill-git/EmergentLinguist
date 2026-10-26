import conllu 
import sys
import json

def extract_ordering(sentence):
    def check_dependents(head_id):
        gen_left, gen_right = 0, 0
        for dep in sentence:
                # First condition: the genitive noun connects via nmod and has Case=Gen
                if dep["head"] == head_id and dep["upos"] == "NOUN" and dep["deprel"] == "nmod" \
                   and dep["feats"] and dep["feats"].get("Case") == "Gen":
                    if int(dep["id"]) < int(head_id):
                        gen_left += 1
                    elif int(dep["id"]) > int(head_id):
                        gen_right += 1

                # Second condition: the genitive noun connects via nmod:poss
                elif dep["head"] == head_id and dep["upos"] == "NOUN" and dep["deprel"] == "nmod:poss":
                    if int(dep["id"]) < int(head_id):
                        gen_left += 1
                    elif int(dep["id"]) > int(head_id):
                        gen_right += 1
                
                # Third condition: the genitive noun connects via nmod and has an ADP dependent with Case=Gen and the case deprel
                elif dep["head"] == head_id and dep["upos"] == "NOUN" and dep["deprel"] == "nmod" and \
                   any([x["head"] == dep["id"] and x["deprel"] == "case" and x["feats"] and 
                        x["feats"].get("Case") == "Gen" for x in sentence]):
                    if int(dep["id"]) < int(head_id):
                        gen_left += 1
                    elif int(dep["id"]) > int(head_id):
                        gen_right += 1
        
        return gen_left, gen_right

    total_gen_left, total_gen_right = 0, 0

    for tok in sentence:
        if tok["upos"] == "NOUN":
            curr_id = tok["id"]
            curr_gen_left, curr_gen_right = check_dependents(curr_id)
            
            total_gen_left += curr_gen_left
            total_gen_right += curr_gen_right
    
    return total_gen_left, total_gen_right
    

def get_final_answer(orderings):
    if all([x == 0 for x in orderings.values()]):
        return "No relevant examples"
    
    dominant = max(orderings.items(), key=lambda x: x[1])

    return dominant[0]
    
    
filepath = sys.argv[1]
out_path = sys.argv[2]
distrib = {"GenN": 0, "NGen": 0}

with open(filepath, "r", encoding="utf-8") as rf:
    conllu_sents = conllu.parse(rf.read())

for conllu_sent in conllu_sents:
    curr_sent_gen_left, curr_sent_gen_right = extract_ordering(conllu_sent)

    distrib["GenN"] += curr_sent_gen_left
    distrib["NGen"] += curr_sent_gen_right

# export result as json
result = {
    "distribution": distrib,
    "final_answer": get_final_answer(distrib)
}

with open(out_path, "w", encoding="utf-8") as wf:
    json.dump(result, wf)
