import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import copy

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

def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def t5_model(input_sentence, best_key):

    model.load_state_dict(torch.load('t5_model_' + best_key + '.pth'))
    model.eval()

    output = []
    for test_input in input_sentence:
        test_input_tokens = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(test_input_tokens.input_ids, max_length=50, num_beams=5, early_stopping=True)
            predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output.append(predicted_text)

    return output

to_solve_demo = """f_past
 f_verb
  d_they
  d_run

f_past
 f_verb
  d_we
  d_sing

f_by
 f_past
  f_verb3
   d_john
   d_kill
   d_mary
 f_prep
  f_noun
   f_past
    f_verb
     d_john
     d_throw
   d_rock
  d_at
  d_mary

f_prep
 f_noun
  f_past
   f_verb
    d_john
    d_push
  d_table
 d_to
 d_wall

f_past
 d_i-am-happy"""

def process(to_solve):
    output = []

    to_solve_new = []
    for item in to_solve.split("\n\n"):
        to_solve_new.append(item)

    for item in to_solve_new:
        print(string_equation(item))
    print()

    def find_largest(equation_list):
        largest = {"past": 0, "verb": 0, "by": 0, "verb3": 0, "noun": 0, "prep": 0, "ask": 0, "poss": 0}
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
        
        process_output = t5_model(dfs_list, best_key)
        
        for i in range(len(process_output)):
            process_output[i] = "d_" + process_output[i].replace(" ", "-")

        for i in range(len(dfs_list)):
            for j in range(len(to_solve_new)):
                to_solve_new[j] = str_form(replace(tree_form(to_solve_new[j]), tree_form(dfs_output[i]), tree_form(process_output[i])))

    for item in to_solve_new:
        print(tree_form(item).name[2:].replace("-", " "))
        output.append(tree_form(item).name[2:].replace("-", " "))

    return output

