import json
import random
import copy
import string

##########################################
# Token generator (controlled random)
##########################################

def gen_token():
    letters = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,6)))
    digit = str(random.randint(0,9))
    return letters + digit
def gen_list(size=10):
    return [gen_token() for _ in range(size)]

##########################################
# Rule functions (30 levels)
##########################################

def L1_sort_alpha(L): return sorted(L)

def L2_reverse(L): return L[::-1]

def L3_sort_len_asc(L): return sorted(L, key=lambda x: len(x))

def L4_sort_len_desc(L): return sorted(L, key=lambda x: -len(x))

def L5_rotate_right(L): return [L[-1]] + L[:-1]

def L6_rotate_left2(L): return L[2:] + L[:2]

def L7_len_ge4(L): return [x for x in L if len(x) >= 4]

def L8_remove_vowel_start(L):
    vowels = set("aeiou")
    return [x for x in L if x[0] not in vowels]

def L9_keep_even_idx(L): return [x for i,x in enumerate(L) if i%2==0]

def L10_min_size_guard(L):
    return L + L if len(L) < 4 else L

def L11_append_index(L):
    return [f"{x}_{i}" for i,x in enumerate(L)]

def L12_sort_by_suffix(L):
    return sorted(L, key=lambda x: int(x.split("_")[-1]))

def L13_remove_suffix(L):
    return [x.split("_")[0] for x in L]

def L14_double_first_letter(L):
    return [x[0]*2 for x in L]

def L15_group_pairs(L):
    out=[]
    for i in range(0,len(L),2):
        out.append(L[i:i+2])
    return out

def L16_reverse_inner(L):
    return [sub[::-1] for sub in L]

def L17_reverse_outer(L):
    return L[::-1]

def L18_flatten(L):
    return [item for sub in L for item in sub]


def L19_prev_word(L):
    out=[]
    for i in range(len(L)):
        if i==0:
            out.append("START")
        else:
            out.append(L[i-1])
    return out

def L20_next_word(L):
    out=[]
    for i in range(len(L)):
        if i==len(L)-1:
            out.append("END")
        else:
            out.append(L[i+1])
    return out

def L21_length_compare(L):
    out=[]
    for i,x in enumerate(L):
        if i==0:
            out.append("SHORT")
        else:
            out.append("LONG" if len(x)>len(L[i-1]) else "SHORT")
    return out

def L22_even_reverse_else_sort(L):
    return L[::-1] if len(L)%2==0 else sorted(L)

def L23_majority_duplicate_remove(L):
    counts={}
    for x in L:
        counts[x]=counts.get(x,0)+1
    if max(counts.values()) > len(L)/2:
        return list(dict.fromkeys(L))
    return L

def L24_conditional_duplicate(L):
    if sorted(L)[0] < sorted(L)[-1]:
        return L + L
    return L[:-1]

def L25_encode_word_index(L):
    return [(x,i) for i,x in enumerate(L)]

def L26_encode_with_length(L):
    return [(x,i,len(x)) for (x,i) in L]

def L27_drop_word_keep_meta(L):
    return [(i,length) for (_,i,length) in L]

def L28_index_only(L):
    return [i for i,_ in enumerate(L)]

def L29_index_parity(L):
    return [(i, i%2) for i in L]

def L30_structure_only(L):
    return list(range(len(L)))

##########################################
# Rule registry
##########################################

LEVEL_RULES = {
 1: ("SORT_ALPHA","Sort alphabetically.",L1_sort_alpha),
 2: ("REVERSE","Reverse the list.",L2_reverse),
 3: ("SORT_LEN_ASC","Sort by length ascending.",L3_sort_len_asc),
 4: ("SORT_LEN_DESC","Sort by length descending.",L4_sort_len_desc),
 5: ("ROTATE_RIGHT","Rotate right by 1.",L5_rotate_right),
 6: ("ROTATE_LEFT2","Rotate left by 2.",L6_rotate_left2),
 7: ("LEN_GE4","Keep words length >=4.",L7_len_ge4),
 8: ("NO_VOWEL_START","Remove words starting with vowel.",L8_remove_vowel_start),
 9: ("EVEN_IDX","Keep even indices.",L9_keep_even_idx),
 10: ("SIZE_GUARD","If size<4 duplicate list.",L10_min_size_guard),
 11: ("APPEND_IDX","Append index suffix.",L11_append_index),
 12: ("SORT_SUFFIX","Sort by numeric suffix.",L12_sort_by_suffix),
 13: ("REMOVE_SUFFIX","Remove index suffix.",L13_remove_suffix),
 14: ("DOUBLE_FIRST","Replace with doubled first letter.",L14_double_first_letter),
 15: ("GROUP_PAIRS","Group into pairs.",L15_group_pairs),
 16: ("REV_INNER","Reverse inner lists.",L16_reverse_inner),
 17: ("REV_OUTER","Reverse outer list.",L17_reverse_outer),
 18: ("FLATTEN","Flatten one level.",L18_flatten),
 19: ("PREV_WORD","Replace with previous word.",L19_prev_word),
 20: ("NEXT_WORD","Replace with next word.",L20_next_word),
 21: ("LEN_COMPARE","Compare length with previous.",L21_length_compare),
 22: ("COND_REV_SORT","If even length reverse else sort.",L22_even_reverse_else_sort),
 23: ("MAJ_DUP_REMOVE","If majority duplicate remove duplicates.",L23_majority_duplicate_remove),
 24: ("COND_DUP","Conditional duplicate or remove last.",L24_conditional_duplicate),
 25: ("ENCODE_META","Encode (word,index).",L25_encode_word_index),
 26: ("ENCODE_LEN","Encode (word,index,length).",L26_encode_with_length),
 27: ("DROP_WORD","Keep (index,length).",L27_drop_word_keep_meta),
 28: ("INDEX_ONLY","Replace with index only.",L28_index_only),
 29: ("INDEX_PARITY","Replace with (index,parity).",L29_index_parity),
 30: ("STRUCT_ONLY","Return indices 0..N-1.",L30_structure_only),
}

##########################################
# Dataset generator
##########################################

def generate_dataset(num_tasks=10, base_size=10, out_file="text_agentic_30lvl.json"):
    dataset=[]
    for tid in range(num_tasks):
        random.seed(tid)  # reproducible randomness
        current=gen_list(base_size)
        levels=[]
        for lvl in range(1,31):
            name,text,fn = LEVEL_RULES[lvl]
            inp=copy.deepcopy(current)
            out=fn(inp)
            levels.append({
                "level":lvl,
                "rule_name":name,
                "rule_text":text,
                "input_list":inp,
                "target_list":out
            })
            current=out
        dataset.append({"task_id":tid,"levels":levels})
    with open(out_file,"w") as f:
        json.dump(dataset,f,indent=2)
    print("Dataset saved:",out_file)

##########################################
# Prompt builder
##########################################

def build_prompt(level_data):
    return f"""
LEVEL {level_data['level']}

RULE:
{level_data['rule_text']}

INPUT LIST:
{level_data['input_list']}

TASK:
Apply the rule exactly.
Return ONLY the resulting list as a Python list.
No explanation. No text. Only the list.
"""

##########################################
# Eval
##########################################

def check_answer(model_list, target_list):
    return model_list == target_list

def evaluate_rollout(model_outputs, task):
    passed=0
    for lvl,model_out in zip(task["levels"],model_outputs):
        if check_answer(model_out, lvl["target_list"]):
            passed+=1
        else:
            break
    return passed

##########################################
# Run
##########################################

if __name__=="__main__":
    generate_dataset(num_tasks=5)
