pos_dic = {"verb": ["kill"], "actor": ["john", "mary", "you"], "noun": ["real-name", "name", "age"], "adj": ["happy"]}
vocab = {"kill": ["killed"], "you": ["your"]}
number = {"0": "verb", "1": "actor", "2": "noun", "3": "adj"}
sentence = "are you happy"
correct = ["'s", "what", "is", "are"]
hypen_word = ["real name"]
eq_list = """f_past
 f_verb
  actor_0
  verb_0

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

import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')
sentence = sentence.replace("'s", " 's")
for item in hypen_word:
    sentence = sentence.replace(item, item.replace(" ", "-"))
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
print("usable words derieved from the sentence: " + ", ".join(new_sentence))

for key in pos_dic.keys():
  for i in range(len(pos_dic[key])-1,-1,-1):
    if pos_dic[key][i] not in new_sentence:
      pos_dic[key].pop(i)

def max_key(equation, key):
  count = 0
  while True:
    if key + "_" + str(count) not in equation:
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
  for item in itertools.product(l_verb, l_actor, l_noun, l_adj):
    new_eq = equation
    for i in range(len(item)):
      for j in range(len(item[i])):
        new_eq = new_eq.replace(number[str(i)] + "_" + str(j), "d_" + item[i][j])
    if "d_none" not in new_eq:
        input_arr.append(new_eq)
import read_8
input_arr = "\n\n".join(input_arr)
print("\nguessed equations:\n")
output_arr = read_8.process(input_arr)
sentence = " ".join(sentence)
input_arr = input_arr.split("\n\n")
print("\nfinal answer: ")
for i in range(len(output_arr)):
  if sentence.replace("-"," ").replace(" 's", "'s") == output_arr[i]:
    print(string_equation(input_arr[i]))
