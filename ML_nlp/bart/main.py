import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

pos_dic = {"verb": ["kill", "throw", "give"], "actor": ["john", "mary", "you", "i"], "noun": ["attention", "rock", "real-name", "name", "age"], "adj": ["happy"], "prep": ["at", "to"]}
vocab = {"kill": ["killed"], "you": ["your"], "i": ["me"], "throw": ["throwing", "threw", "throws"], "give": ["gave"]}
number = {"0": "verb", "1": "actor", "2": "noun", "3": "adj", "4": "prep"}
sentence = "you gave your attention to me"
correct = ["'s", "what", "is", "are", "a"]
hyphen_word = ["real name"]
eq_list = """f_past
 f_verb
  actor_0
  verb_0

f_verb
 actor_0
 verb_0

f_prep
 f_past
  f_noun
   f_verb
    actor_0
    verb_0
  f_poss
   actor_1
   noun_0
 prep_0
 actor_2

f_prep
 f_noun
  f_past
   f_verb
    actor_0
    verb_0
  noun_0
 prep_0
 actor_1

f_adj
 actor_0
 adj_0

f_ask
 f_poss
  actor_0
  noun_0

f_ask
 f_adj
  actor_0
  adj_0

f_past
 f_verb3
  actor_0
  verb_0
  actor_1"""

class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") 
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') 
        node_name = line.strip() 
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] 

def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) 
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) 
        return result
    return recursive_str(node)

def replace(equation, find, r):
    if str_form(equation) == str_form(find):
        return r
    col = TreeNode(equation.name, [])
    for child in equation.children:
        col.children.append(replace(child, find, r))
    return col

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def bart_model(input_sentence, best_key):
    model.load_state_dict(torch.load('bart_model_' + best_key + '.pth'))
    model.eval()

    output = []
    for test_input in input_sentence:
        test_input_tokens = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(test_input_tokens.input_ids, max_length=50, num_beams=5, early_stopping=True)
            predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output.append(predicted_text)

    return output

def process(to_solve):
    output = []

    to_solve_new = []
    for item in to_solve.split("\n\n"):
        to_solve_new.append(item)

    for item in to_solve_new:
        print(string_equation(item))
    print()

    def find_largest(equation_list):
        largest = {"past": 0, "verb": 0, "by": 0, "verb3": 0, "noun": 0, "prep": 0, "ask": 0, "poss": 0, "adj": 0}
        for eq in equation_list:
            largest[tree_form(eq).name[2:]] += 1
        best_score = -999
        best_key = None
        for item in largest.keys():
            if best_score < largest[item]:
                best_score = largest[item]
                best_key = item
        return best_key

    while not all(tree_form(x).name[:2] == "d_" for x in to_solve_new):
        dfs_output = []

        def last_dfs_list(equation):
            if equation.name[:2] != "d_" and all(child.name[:2] == "d_" for child in equation.children):
                dfs_output.append(str_form(equation))
                return
            for child in equation.children:
                last_dfs_list(child)

        for item in to_solve_new:
            last_dfs_list(tree_form(item))

        dfs_output = list(set(dfs_output))

        best_key = find_largest(dfs_output)
        dfs_output = [item for item in dfs_output if tree_form(item).name[2:] == best_key]

        dfs_list = []
        for item in dfs_output:
            if tree_form(item).name[2:] == best_key:
                label = []
                for child in tree_form(item).children:
                    label.append(child.name[2:].replace("-", " "))
                label = " + ".join(label)
                dfs_list.append(label)
        
        process_output = bart_model(dfs_list, best_key)
        
        for i in range(len(process_output)):
            process_output[i] = "d_" + process_output[i].replace(" ", "-")

        for i in range(len(dfs_list)):
            for j in range(len(to_solve_new)):
                to_solve_new[j] = str_form(replace(tree_form(to_solve_new[j]), tree_form(dfs_output[i]), tree_form(process_output[i])))

    for item in to_solve_new:
        print(tree_form(item).name[2:].replace("-", " "))
        output.append(tree_form(item).name[2:].replace("-", " "))

    return output

def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "("
    s = equation_tree.name[2:] + s
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + ","
    s = s[:-1] + ")"
    return s

def string_equation(eq): 
    eq = eq.replace("v_0", "x")
    eq = eq.replace("v_1", "y")
    eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")
    
    return string_equation_helper(tree_form(eq))

import copy
change = []
for key in vocab.keys():
  change += vocab[key]
for key in pos_dic.keys():
  correct += pos_dic[key]

sentence = sentence.split()
new_sentence = []
for item in sentence:
  if item in correct:
    new_sentence.append(item)
  elif item in change:
    for key in vocab.keys():
      if item in vocab[key]:
        new_sentence.append(key)

print("inputted sentence: " + " ".join(sentence))
print("words of processed sentence: " + ", ".join(new_sentence))

for key in pos_dic.keys():
  for i in range(len(pos_dic[key])-1,-1,-1):
    if pos_dic[key][i] not in new_sentence:
      pos_dic[key].pop(i)

def max_key(equation, key):
  count = 0
  while True:
    if key + "_" + str(count) not in equation:
      if count == 0:
          return 1
      return count
    count += 1

import itertools

input_arr = []

def fix_empty_list(arr, max_key_val):
    while len(arr) < max_key_val:
        arr += ["none"]
    return arr

for equation in eq_list.split("\n\n"):
  l_verb = list(itertools.product(fix_empty_list(pos_dic["verb"], max_key(equation, "verb")), repeat=max_key(equation, "verb")))
  l_actor = list(itertools.product(fix_empty_list(pos_dic["actor"], max_key(equation, "actor")), repeat=max_key(equation, "actor")))
  l_noun = list(itertools.product(fix_empty_list(pos_dic["noun"], max_key(equation, "noun")), repeat=max_key(equation, "noun")))
  l_adj = list(itertools.product(fix_empty_list(pos_dic["adj"], max_key(equation, "adj")), repeat=max_key(equation, "adj")))
  l_prep = list(itertools.product(fix_empty_list(pos_dic["prep"], max_key(equation, "prep")), repeat=max_key(equation, "prep")))
  for item in itertools.product(l_verb, l_actor, l_noun, l_adj, l_prep):
    #print(item)
    new_eq = equation
    for i in range(len(item)):
      for j in range(len(item[i])):
        new_eq = new_eq.replace(number[str(i)] + "_" + str(j), "d_" + item[i][j])
    #print(new_eq)
    if "d_none" not in new_eq:
        input_arr.append(new_eq)
        
input_arr = "\n\n".join(input_arr)
print("\nguessed equations:\n")

output_arr = process(input_arr)
sentence = " ".join(sentence)
input_arr = input_arr.split("\n\n")
print("\nfinal answer: ")
for i in range(len(output_arr)):
  if sentence.replace("-"," ").replace(" 's", "'s") == output_arr[i]:
    print(string_equation(input_arr[i]))
