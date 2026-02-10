import json
import random
import math
import copy

##########################################
# Rule functions (28 levels)
##########################################

def L1_add1(M): return [[x+1 for x in row] for row in M]
def L2_sub1(M): return [[x-1 for x in row] for row in M]
def L3_mul2(M): return [[x*2 for x in row] for row in M]
def L4_add5(M): return [[x+5 for x in row] for row in M]
def L5_add_row_index(M): return [[x+i for x in row] for i,row in enumerate(M)]
def L6_add_col_index(M): return [[x+j for j,x in enumerate(row)] for row in M]
def L7_parity(M): return [[x+1 if x%2==0 else x-1 for x in row] for row in M]
def L8_threshold(M): return [[1 if x>5 else 0 for x in row] for row in M]
def L9_transpose(M): return [list(row) for row in zip(*M)]
def L10_hmirror(M): return [row[::-1] for row in M]
def L11_vmirror(M): return M[::-1]
def L12_rotate(M): return [list(row) for row in zip(*M[::-1])]

def L13_row_norm(M):
    out=[]
    for row in M:
        s=sum(row) or 1
        out.append([x//s for x in row])
    return out

def L14_col_norm(M):
    cols=list(zip(*M))
    sums=[sum(c) or 1 for c in cols]
    return [[M[i][j]//sums[j] for j in range(len(M[0]))] for i in range(len(M))]

def L15_row_sum(M): return [[sum(row)] for row in M]
def L16_col_sum(M): return [[sum(col) for col in zip(*M)]]
def L17_diag(M): return [[M[i][i]] for i in range(min(len(M),len(M[0])))]
def L18_anti_diag(M):
    n=min(len(M),len(M[0]))
    return [[M[i][n-i-1]] for i in range(n)]

def L19_neighbor_sum(M):
    h,w=len(M),len(M[0])
    out=[[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            s=0
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni,nj=i+di,j+dj
                if 0<=ni<h and 0<=nj<w:
                    s+=M[ni][nj]
            out[i][j]=s
    return out

def L20_neighbor_mean(M):
    h,w=len(M),len(M[0])
    out=[[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            vals=[]
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni,nj=i+di,j+dj
                if 0<=ni<h and 0<=nj<w:
                    vals.append(M[ni][nj])
            out[i][j]=sum(vals)//len(vals) if vals else 0
    return out

def is_prime(x):
    if x<2: return False
    for i in range(2,int(math.sqrt(x))+1):
        if x%i==0: return False
    return True

def L21_edges(M):
    h,w=len(M),len(M[0])
    out=copy.deepcopy(M)
    for i in range(h):
        for j in range(w):
            if i==0 or j==0 or i==h-1 or j==w-1:
                out[i][j]+=1
    return out

def L22_prime_mask(M):
    return [[x+1 if is_prime(x) else x for x in row] for row in M]

def L23_conditional_structure(M):
    r=sum(map(sum,M))
    c=sum(map(sum, zip(*M)))
    return L9_transpose(M) if r>c else L12_rotate(M)

def L24_comp(M): return L1_add1(L9_transpose(M))

def L25_state_rule(M):
    total=sum(map(sum,M))
    return L12_rotate(M) if total%2==0 else L10_hmirror(M)

def L26_block_sum(M):
    h,w=len(M),len(M[0])
    out=[]
    for i in range(0,h,2):
        row=[]
        for j in range(0,w,2):
            s=0
            for di in [0,1]:
                for dj in [0,1]:
                    if i+di<h and j+dj<w:
                        s+=M[i+di][j+dj]
            row.append(s)
        out.append(row)
    return out

def L27_encode(M):
    return [[(x,i,j) for j,x in enumerate(row)] for i,row in enumerate(M)]

def L28_structure_only(M):
    h,w=len(M),len(M[0])
    return [[(i,j) for j in range(w)] for i in range(h)]

###########################################
# Rule registry
###########################################

LEVEL_RULES = {
  1: ("ADD_1", "Add 1 to every element.", L1_add1),
  2: ("SUB_1", "Subtract 1 from every element.", L2_sub1),
  3: ("MUL_2", "Multiply every element by 2.", L3_mul2),
  4: ("ADD_5", "Add 5 to every element.", L4_add5),
  5: ("ADD_ROW_IDX", "Add the row index to each value in that row.", L5_add_row_index),
  6: ("ADD_COL_IDX", "Add the column index to each value in that column.", L6_add_col_index),
  7: ("PARITY", "If even add 1, if odd subtract 1.", L7_parity),
  8: ("THRESHOLD", "If value > 5 set to 1, else set to 0.", L8_threshold),
  9: ("TRANSPOSE", "Transpose the matrix.", L9_transpose),
  10: ("HMIRROR", "Reverse each row.", L10_hmirror),
  11: ("VMIRROR", "Reverse the order of rows.", L11_vmirror),
  12: ("ROTATE", "Rotate the matrix 90 degrees clockwise.", L12_rotate),
  13: ("ROW_NORM", "Divide each row by its row sum (integer floor).", L13_row_norm),
  14: ("COL_NORM", "Divide each column by its column sum (integer floor).", L14_col_norm),
  15: ("ROW_SUM", "Replace each row with its sum.", L15_row_sum),
  16: ("COL_SUM", "Replace each column with its sum.", L16_col_sum),
  17: ("DIAG", "Extract the main diagonal.", L17_diag),
  18: ("ANTI_DIAG", "Extract the anti-diagonal.", L18_anti_diag),
  19: ("NEIGHBOR_SUM", "Each cell becomes the sum of its 4-neighbors.", L19_neighbor_sum),
  20: ("NEIGHBOR_MEAN", "Each cell becomes the mean of its neighbors.", L20_neighbor_mean),
  21: ("EDGE_INC", "Add 1 to all edge cells.", L21_edges),
  22: ("PRIME_MASK", "Add 1 only to prime-valued cells.", L22_prime_mask),
  23: ("COND_STRUCT", "If row sum > column sum transpose else rotate.", L23_conditional_structure),
  24: ("COMPOSED", "Transpose then add 1 to all values.", L24_comp),
  25: ("STATE_RULE", "If matrix sum even rotate else mirror.", L25_state_rule),
  26: ("BLOCK_SUM", "Each 2x2 block becomes its sum.", L26_block_sum),
  27: ("ENCODE", "Encode each value as (value,row,col).", L27_encode),
  28: ("STRUCT_ONLY", "Ignore values and output positions only.", L28_structure_only),
}

##########################################
# Starting Matrix
##########################################

def gen_matrix(size, low=0, high=9):
    return [[random.randint(low, high) for _ in range(size)] for _ in range(size)]

###########################################
# Dataset generator
##########################################

def generate_dataset(num_tasks=10, base_size=10, out_file="matrix_agentic_30lvl.json"):
    dataset=[]
    for tid in range(num_tasks):
        size=base_size
        current=gen_matrix(size)
        levels=[]
        for lvl in range(1,29):
            name, text, fn = LEVEL_RULES[lvl]
            inp=copy.deepcopy(current)
            out=fn(inp)
            levels.append({
                "level": lvl,
                "rule_name": name,
                "rule_text": text,
                "input_matrix": inp,
                "target_matrix": out
            })
            current=out
            if lvl%3==0:
                size=min(size+1,6)
        dataset.append({"task_id": tid, "levels": levels})
    with open(out_file,"w") as f:
        json.dump(dataset,f,indent=2)
    print("Dataset saved:", out_file)

##########################################
# Prompt builder
##########################################

def build_prompt(level_data):
    return f"""
LEVEL {level_data['level']}

RULE:
{level_data['rule_text']}

INPUT MATRIX:
{level_data['input_matrix']}

TASK:
Apply the rule exactly.
Return ONLY the resulting matrix as a Python list of lists.
No explanation. No text. Only the matrix.
"""

###########################################
# Eval
###########################################

def check_answer(model_matrix, target_matrix):
    return model_matrix == target_matrix

def evaluate_rollout(model_outputs, task):
    passed=0
    for lvl, model_out in zip(task["levels"], model_outputs):
        if check_answer(model_out, lvl["target_matrix"]):
            passed+=1
        else:
            break
    return passed

###########################################
# Running 
##########################################

if __name__=="__main__":
    generate_dataset(num_tasks=5) #
