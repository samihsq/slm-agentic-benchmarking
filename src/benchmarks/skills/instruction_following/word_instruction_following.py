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

def L3_sort_len_asc(L): return sorted(L, key=lambda x: (len(x), x))

def L4_sort_len_desc(L): return sorted(L, key=lambda x: (-len(x), x))

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
    return sorted(L, key=lambda x: (int(x.split("_")[-1]), x))

def L13_remove_suffix(L):
    return [x.rsplit("_", 1)[0] for x in L]

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
            if len(x) > len(L[i-1]):
                out.append("LONGER")
            elif len(x) < len(L[i-1]):
                out.append("SHORTER")
            else:
                out.append("EQUAL")
    return out

def L22_even_reverse_else_sort(L):
    return L[::-1] if len(L)%2==0 else sorted(L)

def L23_majority_duplicate_remove(L):
    if not L:
        return L
    counts={}
    for x in L:
        counts[x]=counts.get(x,0)+1
    if max(counts.values()) > len(L)//2:
        seen = set()
        result = []
        for x in L:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
    return L

def L24_conditional_duplicate(L):
    if not L:
        return L
    if sorted(L)[0] < sorted(L)[-1]:
        return L + L
    return L[:-1]

def L25_encode_word_index(L):
    return [[x,i] for i,x in enumerate(L)]

def L26_encode_with_length(L):
    return [[x,i,len(x)] for x,i in L]

def L27_drop_word_keep_meta(L):
    return [[i,length] for _,i,length in L]

def L28_index_only(L):
    return [pair[0] for pair in L]

def L29_index_parity(L):
    return [[i, i%2] for i in L]

def L30_structure_only(L):
    return list(range(len(L)))

##########################################
# Rule registry
##########################################

LEVEL_RULES = {
 1: ("SORT_ALPHA",
     "Sort the list alphabetically (a-z).",
     L1_sort_alpha),
 2: ("REVERSE",
     "Reverse the order of the list.",
     L2_reverse),
 3: ("SORT_LEN_ASC",
     "Sort by string length ascending. Break ties alphabetically.",
     L3_sort_len_asc),
 4: ("SORT_LEN_DESC",
     "Sort by string length descending. Break ties alphabetically.",
     L4_sort_len_desc),
 5: ("ROTATE_RIGHT",
     "Rotate the list right by 1 position (last element moves to front).",
     L5_rotate_right),
 6: ("ROTATE_LEFT2",
     "Rotate the list left by 2 positions (first two elements move to end).",
     L6_rotate_left2),
 7: ("LEN_GE4",
     "Keep only words with length >= 4. Preserve order.",
     L7_len_ge4),
 8: ("NO_VOWEL_START",
     "Remove words that start with a vowel (a, e, i, o, u). Preserve order.",
     L8_remove_vowel_start),
 9: ("EVEN_IDX",
     "Keep only elements at 0-based even indices (0, 2, 4, ...). Preserve order.",
     L9_keep_even_idx),
 10: ("SIZE_GUARD",
      "If the list has fewer than 4 elements, concatenate it with itself (double it). Otherwise return it unchanged.",
      L10_min_size_guard),
 11: ("APPEND_IDX",
      "Append a 0-based index suffix to each word: word_0, word_1, word_2, etc.",
      L11_append_index),
 12: ("SORT_SUFFIX",
      "Sort by the numeric suffix after the underscore (ascending). Break ties alphabetically by full string.",
      L12_sort_by_suffix),
 13: ("REMOVE_SUFFIX",
      "Remove the last underscore and everything after it from each word (e.g. 'abc_3' becomes 'abc').",
      L13_remove_suffix),
 14: ("DOUBLE_FIRST",
      "Replace each word with its first character repeated twice (e.g. 'hello' becomes 'hh').",
      L14_double_first_letter),
 15: ("GROUP_PAIRS",
      "Group consecutive elements into sublists of 2. If the list has odd length, the last sublist has 1 element.",
      L15_group_pairs),
 16: ("REV_INNER",
      "Reverse each inner sublist. The outer list order stays the same.",
      L16_reverse_inner),
 17: ("REV_OUTER",
      "Reverse the order of the outer list. Inner sublists stay unchanged.",
      L17_reverse_outer),
 18: ("FLATTEN",
      "Flatten one level of nesting: each inner sublist's elements become top-level elements.",
      L18_flatten),
 19: ("PREV_WORD",
      "Replace each element with the element before it. The first element becomes \"START\".",
      L19_prev_word),
 20: ("NEXT_WORD",
      "Replace each element with the element after it. The last element becomes \"END\".",
      L20_next_word),
 21: ("LEN_COMPARE",
      "Compare each element's length to the previous element's length. Output \"LONGER\" if longer, \"SHORTER\" if shorter, \"EQUAL\" if same. The first element is always \"SHORT\".",
      L21_length_compare),
 22: ("COND_REV_SORT",
      "If the list has even length, reverse it. If odd length, sort it alphabetically.",
      L22_even_reverse_else_sort),
 23: ("MAJ_DUP_REMOVE",
      "If any single value appears more than half the list's length (floor division), remove all duplicates keeping only the first occurrence of each. Otherwise return the list unchanged.",
      L23_majority_duplicate_remove),
 24: ("COND_DUP",
      "Sort the list alphabetically. If the first element is less than the last element, concatenate the original (unsorted) list with itself. Otherwise, remove the last element from the original list.",
      L24_conditional_duplicate),
 25: ("ENCODE_META",
      "Replace each element with a [word, index] pair, where index is the 0-based position. Output a list of 2-element lists.",
      L25_encode_word_index),
 26: ("ENCODE_LEN",
      "Each element is currently [word, index]. Extend it to [word, index, length] where length is len(word). Output a list of 3-element lists.",
      L26_encode_with_length),
 27: ("DROP_WORD",
      "Each element is currently [word, index, length]. Drop the word and keep [index, length]. Output a list of 2-element lists.",
      L27_drop_word_keep_meta),
 28: ("INDEX_ONLY",
      "Each element is currently [index, length]. Extract just the index (first element) from each pair. Output a flat list of integers.",
      L28_index_only),
 29: ("INDEX_PARITY",
      "Each element is an integer. Replace it with [value, value % 2]. Output a list of 2-element lists.",
      L29_index_parity),
 30: ("STRUCT_ONLY",
      "Return a list of integers from 0 to N-1, where N is the current list length.",
      L30_structure_only),
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
