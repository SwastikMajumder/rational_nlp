# Copyright (c) 2024 Swastik Majumder
# All rights reserved.

# Part of the book Transcendental Computing with Python: Applications in Mathematics - Edition 1

from collections import deque
import copy
file_content_1 = None
file_content_2 = None
file_content_3 = None
file_content_4 = None
file_content_5 = None
file_content_6 = None
file_content_7 = None
file_content_8 = None
file_content_9 = None
file_content_10 = None
file_content_11 = None
file_content_12 = None
file_content_13 = None
with open("geo_t/formula_list_1.txt", 'r') as file:
  file_content_1 = file.read()
with open("geo_t/formula_list_2.txt", 'r') as file:
  file_content_2 = file.read()
with open("geo_t/formula_list_3.txt", 'r') as file:
  file_content_3 = file.read()
with open("geo_t/formula_list_4.txt", 'r') as file:
  file_content_4 = file.read()
with open("geo_t/formula_list_5.txt", 'r') as file:
  file_content_5 = file.read()
with open("geo_t/formula_list_6.txt", 'r') as file:
  file_content_6 = file.read()
with open("geo_t/formula_list_7.txt", 'r') as file:
  file_content_7 = file.read()
with open("geo_t/formula_list_8.txt", 'r') as file:
  file_content_8 = file.read()
with open("geo_t/formula_list_9.txt", 'r') as file:
  file_content_9 = file.read()
with open("geo_t/formula_list_10.txt", 'r') as file:
  file_content_10 = file.read()
with open("geo_t/formula_list_11.txt", 'r') as file:
  file_content_11 = file.read()
with open("geo_t/formula_list_12.txt", 'r') as file:
  file_content_12 = file.read()
with open("geo_t/formula_list_13.txt", 'r') as file:
  file_content_13 = file.read()
def access_file(num):
  global file_content_1
  global file_content_2
  global file_content_3
  global file_content_4
  global file_content_5
  global file_content_6
  global file_content_7
  global file_content_8
  global file_content_9
  global file_content_10
  global file_content_11
  global file_content_12
  global file_content_13
  if num == 1:
    return file_content_1
  elif num == 2:
    return file_content_2
  elif num == 3:
    return file_content_3
  elif num == 4:
    return file_content_4
  elif num == 5:
    return file_content_5
  elif num == 6:
    return file_content_6
  elif num == 7:
    return file_content_7
  elif num == 8:
    return file_content_8
  elif num == 9:
    return file_content_9
  elif num == 10:
    return file_content_10
  elif num == 11:
    return file_content_11
  elif num == 12:
    return file_content_12
  elif num == 13:
    return file_content_13
# Basic data structure, which can nest to represent math equations
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

# convert string representation into tree
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") # add a dummy node
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') # count the spaces, which is crucial information in a string representation
        node_name = line.strip() # remove spaces, when putting it in the tree form
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] # remove dummy node

# convert tree into string representation
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) # spacings
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) # one node in one line
        return result
    return recursive_str(node)

# Generate transformations of a given equation provided only one formula to do so
# We can call this function multiple times with different formulas, in case we want to use more than one
# This function is also responsible for computing arithmetic, pass do_only_arithmetic as True (others param it would ignore), to do so
def apply_individual_formula_on_given_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic=False, structure_satisfy=False):
    variable_list = {}
    
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs):
        nonlocal variable_list
        # u can accept anything and p is expecting only integers
        # if there is variable in the formula
        if node_type(formula_lhs.name) in {"u_", "p_"}: 
            if formula_lhs.name in variable_list.keys(): # check if that variable has previously appeared or not
                return str_form(variable_list[formula_lhs.name]) == str_form(equation) # if yes, then the contents should be same
            else: # otherwise, extract the data from the given equation
                if node_type(formula_lhs.name) == "p_" and "v_" in str_form(equation): # if formula has a p type variable, it only accepts integers
                    return False
                variable_list[formula_lhs.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula_lhs.name or len(equation.children) != len(formula_lhs.children): # the formula structure should match with given equation
            return False
        for i in range(len(equation.children)): # go through every children and explore the whole formula / equation
            if does_given_equation_satisfy_forumla_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    if structure_satisfy:
      return does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs)
    # transform the equation as a whole aka perform the transformation operation on the entire thing and not only on a certain part of the equation
    def formula_apply_root(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] # fill the extracted data on the formula rhs structure
        data_to_return = TreeNode(formula.name, None) # produce nodes for the new transformed equation
        for child in formula.children:
            data_to_return.children.append(formula_apply_root(copy.deepcopy(child))) # slowly build the transformed equation
        return data_to_return
    count_target_node = 1
    # try applying formula on various parts of the equation
    def formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if does_given_equation_satisfy_forumla_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: # if formula lhs structure is satisfied by the equation given
                count_target_node -= 1
                if count_target_node == 0: # and its the location we want to do the transformation on
                    return formula_apply_root(copy.deepcopy(formula_rhs)) # transform
        else: # perform arithmetic
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children): # if only numbers
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:])) # convert string into a number
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(sum(x))) # add all
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item # multiply all
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2: # power should be two or a natural number more than two
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
        if node_type(equation.name) in {"d_", "v_"}: # reached a leaf node
            return equation
        for child in equation.children: # slowly build the transformed equation
            data_to_return.children.append(formula_apply_various_sub_equation(copy.deepcopy(child), formula_lhs, formula_rhs, do_only_arithmetic))
        return data_to_return
    cn = 0
    # count how many locations are present in the given equation
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    transformed_equation_list = []
    count_nodes(equation)
    for i in range(1, cn + 1): # iterate over all location in the equation tree
        count_target_node = i
        orig_len = len(transformed_equation_list)
        tmp = formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic)
        if str_form(tmp) != str_form(equation): # don't produce duplication, or don't if nothing changed because of transformation impossbility in that location
            transformed_equation_list.append(str_form(tmp)) # add this transformation to our list
    return transformed_equation_list 

# Function to read formula file
def return_formula_file(num):
    content = access_file(num)
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f] # return

# Function to generate neighbor equations
def generate_transformation(equation, num=1):
    input_f, output_f = return_formula_file(num) # load formula file
    transformed_equation_list = []
    #transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) # perform arithmetic
    for i in range(len(input_f)): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) # set list to remove duplications
count_n = 0
def search_4(equation, depth, visited, goal_reached, number, print_option):
    global count_n
    if depth == 0: # limit the search
        return None
    equation = str_form(simply_mul_main(tree_form(equation)))
    if visited is None:
        visited = set()
    if equation in visited:
        return None
    if print_option == 0:
      print(print_equation(str_form(TreeNode("f_int", [TreeNode("f_mul", [tree_form(equation), TreeNode("f_dif", [TreeNode("v_0")])])]))))
    elif print_option == 1:
      print(print_equation(equation))
    count_n += 1
    if (goal_reached and goal_reached(equation)) or count_n > 15:
      return [equation]
    output = list(set(apply_individual_formula_on_given_equation(tree_form(equation), None, None, True))) + generate_transformation(equation, number[1])
    if len(output) > 0:
      output = [output[0]]
    else:
      output = generate_transformation(equation, number[0]) # generate equals to the asked one
      if len(output) == 0:
        output = generate_transformation(equation, 5)
    for i in range(len(output)):
        result = search_4(output[i], depth-1, visited, goal_reached, number, print_option) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    output = list(set(output))
    return output
def search(equation, depth, visited=None, goal_reached=None, number=[1,2], print_option=0):
    global count_n
    count_n= 0
    return search_4(equation, depth, visited, goal_reached, number, print_option)
def search_2(equation, num=6):
  while True:
    output = list(set(apply_individual_formula_on_given_equation(tree_form(equation), None, None, True))) + generate_transformation(equation, num)
    if len(output) == 0:
      return equation
    else:
      equation = output[0]
      #print(print_equation(equation))
def search_3(equation):
  if equation.name == "f_add":
    pass
  else:
    equation = TreeNode("", [equation])
  content = access_file(4)
  x = content.split("\n\n")
  return all(any(apply_individual_formula_on_given_equation(child, tree_form(item), None, False, True) for item in x) for child in equation.children)

# remove unecessary brackets, for fancy printing and spohistication
def flatten_tree(node):
    if not node.children:
        return node
    if node.name in {"f_add", "f_mul"}: # commutative property supporting functions
        merged_children = [] # merge all the children
        for child in node.children:
            flattened_child = flatten_tree(child)
            if flattened_child.name == node.name:
                merged_children.extend(flattened_child.children)
            else:
                merged_children.append(flattened_child)
        return TreeNode(node.name, merged_children)
    else:
        node.children = [flatten_tree(child) for child in node.children]
        return node

# fancy print
def print_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    if len(equation_tree.children) == 1:
        s = equation_tree.name[2:] + s
    sign = {"f_add": "+", "f_and": "∧", "f_or": "∨", "f_imp": "→", "f_bi": "↔", "f_not": "¬", "f_tri": "Δ", "f_angle": "∠", "f_line": "l", "f_eq": "=", "f_para": "||", "f_con": "≅"} # operation symbols
    if equation_tree.name in {"f_tri", "f_angle", "f_line", "f_para"}:
      s += sign[equation_tree.name]
    for child in equation_tree.children:
      if equation_tree.name not in {"f_tri", "f_angle", "f_line", "f_para"}:
        s+= print_equation_helper(child) + sign[equation_tree.name]
      else:
        s += print_equation_helper(child)
    if equation_tree.name not in {"f_tri", "f_angle", "f_line", "f_para"}:
      s = s[:-1] + ")"
    else:
      s += ")"
    return s
n_var = 0
# fancy print main function
def print_equation(eq):
    global n_var
    #eq = str_form(flatten_tree(tree_form(eq)))
    if n_var == 0:
      eq = eq.replace("v_0", "x")
      eq = eq.replace("v_1", "y")
      eq = eq.replace("v_2", "z")
    elif n_var == 1:
      eq = eq.replace("v_0", "y")
      eq = eq.replace("v_1", "x")
      eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")
    return print_equation_helper(tree_form(eq))

q = """f_mul
 v_0
 f_pow
  f_add
   f_pow
    v_0
    d_2
   d_5
  d_17"""
q = """f_sqt
 f_add
  v_0
  d_5"""
q = """f_mul
 v_0
 f_sqt
  f_add
   v_0
   d_-5"""
#print(print_equation(str_form(TreeNode("f_int", [TreeNode("f_mul", [tree_form(q), TreeNode("f_dif", [TreeNode("v_0")])])]))))
def final(q):
  for item in [q]+search(q, 4, None, lambda x: search_3(flatten_tree(tree_form(x))), [1,2], 0):
    if search_3(flatten_tree(tree_form(item))):
      print("found")
      return item
  return None

def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col

def break_equation(equation):
    sub_equation_list = [equation]
    equation = tree_form(equation)
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(str_form(child)) # collect broken equations
    return sub_equation_list

def goal_rhs(equation, eq):
  if str_form(tree_form(equation).children[1]) == eq:
    return True
  return False

def req_algebra(qq, eq="v_1"):
  res = [qq]+search(qq, 4, None, None, [7,8], 1)
  res = [item for item in res if goal_rhs(item, eq)]
  res = sorted(res, key=lambda x: len(x))
  if len(res) >= 1:
    return res[0]
  return None


def simple_int(q, sub_2=None):
  global n_var
  
  output = final(q)
  if output is None:
    for item in break_equation(q):
      if item == q:
        continue
      if tree_form(item).name not in {"f_add", "f_sqt"}:
        continue
      q2 = replace(tree_form(q), tree_form(item), tree_form("v_2"))
      q2 = tree_form(str_form(q2).replace("v_0", "v_1").replace("v_2", "v_0"))
      q2 = TreeNode("f_div", [q2, tree_form(search_2(str_form(TreeNode("f_dif", [tree_form(item)]))).replace("v_0", "v_1"))])
      #n_var += 1
      output = final(str_form(q2))
      if output is None:
        continue
      print(print_equation(str_form(TreeNode("f_int", [TreeNode("f_mul", [tree_form(output), TreeNode("f_dif", [TreeNode("v_0")])])]))), end=" ")
      #n_var -=  1
      print("y="+print_equation(item))
      solve_int(output, [sub_2, item])
      return True
  else:
      print(print_equation(str_form(TreeNode("f_int", [TreeNode("f_mul", [tree_form(output), TreeNode("f_dif", [TreeNode("v_0")])])]))))
      solve_int(output, [sub_2])
      return True
  return False
#print(simple_int(q))

def simply_mul(equation):
  product = 1
  pw = 0
  orig_equation = copy.deepcopy(equation)
  equation = flatten_tree(equation)
  output = TreeNode(equation.name)
  for child in equation.children:
    if child.name[:2] == "d_":
      product *= int(child.name[2:])
    elif child.name == "v_0":
      pw += 1
    elif child.name == "f_pow" and child.children[0].name == "v_0" and child.children[1].name[:2] == "d_":
      pw += int(child.children[1].name[2:])
    else:
      output.children.append(child)
  if product != 1:
    output.children.append(tree_form("d_"+str(product)))
  if pw > 1:
    output.children.append(TreeNode("f_pow", [tree_form("v_0"), tree_form("d_"+str(pw))]))
  elif pw == 1:
    output.children.append(tree_form("v_0"))
  if len(output.children) != 2:
    return orig_equation
  return output
import copy


def simply_mul_main(equation):
  if equation.name == "f_mul" and len(flatten_tree(copy.deepcopy(equation)).children)>2:
    return simply_mul(copy.deepcopy(equation))
  col = TreeNode(equation.name)
  for child in equation.children:
    col.children.append(simply_mul_main(child))
  return col

def bi_int(equation):
  global n_var
  for item in break_equation(equation):
    if item == equation:
      continue
    #if item == "v_0":
    #  continue
    rhs = tree_form(item)
    rhs = TreeNode("f_eq", [tree_form("v_1"), rhs])
    #output = search(str_form(rhs), 4, None, lambda x: tree_form(x).children[1].name == "v_0", [7,8], 1)
    output = req_algebra(str_form(rhs), "v_0")
    if output is None:
      continue
    output = str_form(tree_form(output).children[0])
    output_tmp = str_form(TreeNode("f_dif", [tree_form(output)]))
    output_2 = search_2(output_tmp.replace("v_1", "v_0")).replace("v_0", "v_1")
    final = tree_form(equation)
    final = replace(final, tree_form(item), tree_form("v_1"))
    final = replace(final, tree_form("v_0"), tree_form(output))
    final = TreeNode("f_mul", [final, tree_form(output_2)])

    #final = TreeNode("f_mul", [final, tree_form("v_1")])
    final = str_form(final).replace("v_1", "v_0")
    #n_var += 1
    simple_int(final, item)
    #n_var -= 1

def solve_int(equation, sub=None):
  #return
  equation = str_form(TreeNode("f_int", [tree_form(equation), tree_form("v_0")]))
  equation = search_2(equation, 11)
  #print("HIHI")
  print(print_equation(equation))
  if sub is not None:
    for item in sub:
      if item is None:
        continue
      print(print_equation(item))
      equation = replace(tree_form(equation.replace("v_0", "v_1")), tree_form("v_1"), tree_form(item))
      equation = str_form(equation).replace("v_1", "v_0")
  res = [equation]+search(equation, 12, None, None, [9,10], 1)
  res = sorted(res, key=lambda x: len(x))
  print(print_equation(res[0]))
  
qq = """f_eq
 v_0
 f_sqt
  f_add
   v_1
   d_5"""
#bi_int(q)
import itertools
def search_7(equation, depth, visited):
    if depth == 0: # limit the search
        return None
    if visited is None:
        visited = set()
    if equation in visited:
        return None
    print(print_equation(equation))

    output = []
    et = flatten_tree(tree_form(equation))
    if len(et.children) >= 2 and et.name in {"f_and"}:
      for item1 in itertools.permutations(et.children[:3]):
        for item2 in itertools.permutations(et.children[3:6]):
          for item3 in itertools.permutations(et.children[6:9]):
            for item4 in itertools.permutations(et.children[9:12]):
              equation_p = TreeNode(et.name, list(item1)+list(item2)+list(item3)+list(item4))
              output += generate_transformation(str_form(equation_p), 12)
    output = list(set(output))
    for i in range(len(output)):
        result = search_7(output[i], depth-1, visited) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    output = list(set(output))
    return output

def opposite_angle(formula, answer, equation):
  
  for item in itertools.permutations(list(range(5))):
    ans = answer
    eq_copy = copy.deepcopy(equation)
    for i,x in enumerate(item):
      eq_copy = [y.replace("d_"+chr(ord("A")+i), "p_"+str(x)) for y in eq_copy]
      ans = ans.replace("p_"+str(x), "d_"+chr(ord("A")+i))
    
    #eq_copy = [str_form(x) for x in eq_copy]
    if all(str_form(x) in eq_copy for x in tree_form(formula).children) and ans not in equation:
      #return str_form(TreeNode("f_and", tree_form(equation).children+[tree_form(ans)]))
      return [ans]
  return []

def formula_apply(equation):
  output = []
  for item in equation:
    output += generate_transformation(item, 11)
  output = list(set(output))
  return [item for item in output if item not in equation]

def subs(equation):
  eq_list = [item for item in equation if tree_form(item).name == "f_eq"]
  new = []
  for item in itertools.permutations(eq_list, 2):
    tmp = replace(tree_form(item[0]), tree_form(item[1]).children[0], tree_form(item[1]).children[1])
    if tmp != item[0] and tmp not in eq_list:
      new.append(str_form(tmp))
  return list(set(new))


def line_angle(equation):
  new = []
  line = []
  for item in equation:
    if tree_form(item).name == "f_line":
      line.append(item)
  for item in equation:
    for s in line:
      s = tree_form(s)
      s.name = "f_angle"
      tmp = replace(tree_form(item), s, tree_form("d_180"))
      if tmp not in equation:
        new.append(str_form(tmp))
  #new = [item for item in new if item[:2]!="d_"]
  return list(set(new))

verb = [("run", "runs", "running", "ran"), ("kill", "kills", "killing", "killed"), ("sleep", "sleeps", "sleeping", "slept")]
adjective = ["happy", "honest"]

def ind_pos(sentence, pos, ind=0):
  for i, item in enumerate(sentence):
    if item[ind] == pos:
      return i
  return -1

pronoun_list = [("he", "him"), ("i", "me")]

def because(a, b):
  return a + [("conjunction", "because")] + b

def red_because(a, b):
  word = None
  global pronoun_list
  if b[0][0] == "pronoun":
    for item in pronoun_list:
      if item[0]==b[0][1]:
        word = item[1]
        
  return a + [("conjunction", "because"), ("preposition", "of"), ("pronoun", word)]

def joins(a, b):
  if ind_pos(a, "adjective") != -1 or ind_pos(b, "adjective") != -1:
    return "can't simplify"
  if replace_tense_word([a[-1]], 3) != replace_tense_word([b[-1]], 3) or ind_pos(b, "aux") != 1:
    return "can't simplify"
  word = None
  if b[0][0] == "pronoun":
    for item in pronoun_list:
      if item[0]==b[0][1]:
        word = item[1]
  return a + [("pronoun", word)]

def inter2(sentence):
  output = []
  if ind_pos(sentence, "because", 1) != -1:
    return "can't simplify"
    #a = copy.deepcopy(sentence[:ind_pos(sentence, "because", 1)])
    #print(a)
    #return inter(a) + [("preposition", "after")] + sentence[ind_pos(sentence, "because", 1)+1:]
  if sentence[0][0] == "pronoun":
    output.append(("pronoun", "who"))
  if ind_pos(sentence, "aux") != -1:
    if sentence[ind_pos(sentence, "aux")] != ("aux", "am"):
      output.append(sentence[ind_pos(sentence, "aux")])
    else:
      output.append(("aux", "is"))
  if ind_pos(sentence, "verb") != -1:
    output.append(sentence[ind_pos(sentence, "verb")])
  if sentence[-1] == ("pronoun", "him"):
    output.append(sentence[-1])
  if ind_pos(sentence, "adjective") != -1:
    output.append(sentence[ind_pos(sentence, "adjective")])
  return output

def invert(sentence_orig):
  sentence = copy.deepcopy(sentence_orig)
  if "not" in " ".join(break_s(sentence)[1]):
    return "can't simplify"
  index = ind_pos(sentence, "adjective")
  if index == -1:
    index = ind_pos(sentence, "verb")
  else:
    sentence.insert(index, ("adverb", "not"))
    return sentence
  if index == -1:
    if ind_pos(sentence, "i_pronoun") != -1:
      sentence.insert(ind_pos(sentence, "do"), ("adverb", "not"))
      return sentence
    if ind_pos(sentence, "how", 1) != 1:
      sentence.insert(ind_pos(sentence, "aux")+1, ("adverb", "not"))
      return sentence
    return "can't simplify"
  else:
    sentence.insert(index, ("verb", "not"))
    #print(sentence)
    if sentence == tense(copy.deepcopy(sentence), "past"):
      #print(sentence)
      if sentence == replace_tense_word(sentence, 2):
        return sentence
      sentence.insert(index, ("do", "did"))
      sentence = replace_tense_word(sentence, 0)
    elif sentence == tense(copy.deepcopy(sentence), "present"):
      if sentence == replace_tense_word(sentence, 1):
        sentence.insert(index, ("do", "does"))
      elif sentence == replace_tense_word(sentence, 2):
        return sentence
      else:
        sentence.insert(index, ("do", "do"))
      sentence = replace_tense_word(sentence, 0)
  #print(sentence)
  return sentence

def break_s(sentence):
  if sentence == "can't simplify":
    return ["a", "a"], ["", ""]
  return [item[0] for item in sentence], [item[1] for item in sentence]

def replace_tense_word(sentence, num):
  global verb
  s = break_s(sentence)[1]
  for item in verb:
    for i in range(len(item)):
      if i == num:
        continue
      if item[i] in s:
        s[s.index(item[i])] = item[num]
  return list(zip(break_s(sentence)[0], s))

p_name = {("pronoun", "i"): ("aux", "am"), ("pronoun", "he"): ("aux", "is")}

def inter(sentence):
  output = []
  #sentence = copy.deepcopy(sentence)
  #sentence.remove(("adverb", "away"))
  if ind_pos(sentence, "because", 1) != -1:
    output.append(("i_adverb", "why"))
    es = sentence[:ind_pos(sentence, "because", 1)+1]
    if es == tense(es, "past"):
      if ind_pos(es, "verb") != -1 and es != replace_tense_word(es, 2):
        #print("hi")
        output.append(("do", "did"))
        output.append(es[ind_pos(es, "pronoun")])
        output.append(replace_tense_word(es[ind_pos(es, "verb")], 0)[0])
        output.append(es[-1])
        return output
    if ind_pos(sentence, "adjective") != -1:
      output.append(sentence[ind_pos(sentence, "aux")])
      output.append(sentence[ind_pos(sentence, "pronoun")])
      output.append(sentence[ind_pos(sentence, "adjective")])
      return output
  if ind_pos(sentence, "adjective") != -1:
    output.append(("adverb", "how"))
  if ind_pos(sentence, "verb") != -1:
    output.append(("i_pronoun", "what"))
  if ind_pos(sentence, "verb") != -1:
    if sentence[1][0] == "aux":
      output += [("verb", "happened"), ("preposition", "to")]
      for item in pronoun_list:
        if item[0] == sentence[0][1]:
          output.append(("pronoun", item[1]))
      return output
    if sentence == tense(sentence, "past"):
      output.append(("do", "did"))
    else:
      if sentence[0] == ("pronoun", "i"):
        output.append(("do", "do"))
      else:
        output.append(("do", "does"))
    output.append(sentence[ind_pos(sentence, "pronoun")])
    output.append(("do", "do"))
  else:
    output.append(sentence[ind_pos(sentence, "aux")])
    output.append(sentence[ind_pos(sentence, "pronoun")])
  """
  elif ind_pos(sentence, "verb") != -1:
    output[-1] = replace_tense_word([output[-1]], 3)[0]
  index = ind_pos(sentence, "pronoun")
  if index!=-1:
    output.append(sentence[index])
  if :
    output.append(("do", "doing"))
  """
  return output

def tense(sentence, num):
  global verb
  #if ind_pos(sentence, "away", 1) != -1:
  #  if sentence[-1] == ("adverb", "away"):
  #    return tense(sentence[:-1], num) + [("adverb", "away")]
  if ind_pos(sentence, "because", 1) != -1:
    return "can't simplify"
  if (num=="past" and ind_pos(sentence, "adjective") != -1) or (sentence == replace_tense_word(sentence, 2) and num=="past"):
    if ("aux", "is") in sentence:
      sentence[sentence.index(("aux", "is"))] = ("aux", "was")
    if ("aux", "am") in sentence:
      sentence[sentence.index(("aux", "am"))] = ("aux", "was")
    if ("aux", "are") in sentence:
      sentence[sentence.index(("aux", "are"))] = ("aux", "were")
    #print(sentence)
    return sentence
  if num == "past":
    if ind_pos(sentence, "does", 1)!= -1:
      sentence[sentence.index(("do", "does"))] = ("do", "did")
      return sentence
    elif ind_pos(sentence, "do", 1)!=-1:
      sentence[sentence.index(("do", "do"))] = ("do", "did")
      return sentence
    elif ind_pos(sentence, "did", 1)!=-1:
      return sentence
  if num == "present":
    if sentence == replace_tense_word(sentence, 0):
      num = 0
    elif sentence == replace_tense_word(sentence, 1):
      num = 1
    else:
      num = 2
  elif num == "past":
    num = 3
    if ("aux", "is") in sentence:
      sentence[sentence.index(("aux", "is"))] = ("aux", "was")
  sentence = replace_tense_word(sentence, num)
  #if num==3 and ind_pos(sentence, "not", 1) == -1 and ("aux", "was") in sentence:
  #  sentence.remove(("aux", "was"))
  #print(sentence)
  return sentence
def fx_nest(terminal, fx, depth):
    def neighboring_math_equation(curr_tree, depth=depth): # Generate neighbouring equation trees
        def is_terminal(name):
            return not (name in fx.keys()) # Operations are not leaf nodes
        element = None # What to a append to create something new
        def append_at_last(curr_node, depth): # Append something to generate new equation
            if (is_terminal(element) and depth == 0) or (not is_terminal(element) and depth == 1): # The leaf nodes can't be operations
                return None
            if not is_terminal(curr_node.name):
                if len(curr_node.children) < fx[curr_node.name]: # An operation can take only a mentioned number of arugments
                    curr_node.children.append(TreeNode(element))
                    return curr_node
                for i in range(len(curr_node.children)):
                    output = append_at_last(copy.deepcopy(curr_node.children[i]), depth - 1)
                    if output is not None: # Check if the sub tree has already filled with arugments
                        curr_node.children[i] = copy.deepcopy(output)
                        return curr_node
            return None
        new_math_equation_list = []
        for item in terminal + list(fx.keys()): # Create new math equations with given elements
            element = item # set the element we want to use to create new math equation
            tmp = copy.deepcopy(curr_tree)
            result = append_at_last(tmp, depth)
            if result is not None:
                new_math_equation_list.append(result)
        return new_math_equation_list
    all_possibility = []
    # explore mathematics itself with given elements
    # breadth first search, a widely used algorithm
    def bfs(start_node):
        nonlocal all_possibility
        queue = deque()
        visited = set()
        queue.append(start_node)
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbors = neighboring_math_equation(current_node)
                if neighbors == []:
                    all_possibility.append(str_form(current_node))
                    all_possibility = list(set(all_possibility)) # remove duplicates
                    if len(all_possibility) > 75:
                        return all_possibility
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys(): # use all the elements
        bfs(TreeNode(item))
    return all_possibility # return mathematical equations produce

#fx_nest(["v_0", "v_1"], {"f_invert
store = []
def compute(param, equation):
  global store
  output = []
  equation = copy.deepcopy(equation)
  for item in equation.children:
    if item.name[:2] != "v_":
      tmp = compute(param, item)
      if tmp == "can't simplify":
        return tmp
      else:
        store.append(tmp)
        output.append(tmp)
    else:
      output.append(param[int(item.name[2:])])
  output = copy.deepcopy(output)
  ans = None
  if equation.name == "f_invert":
    ans = invert(*output)
  elif equation.name == "f_past":
    ans = tense(*output, "past")
  elif equation.name == "f_join":
    ans = joins(*output)
  elif equation.name == "f_inter":
    ans = inter(*output)
  elif equation.name == "f_because":
    ans = because(*output)
  elif equation.name == "f_rbecause":
    ans = red_because(*output)
  return copy.deepcopy(ans)

def cont(sentence):
  if ind_pos(sentence, "aux") == -1:
    if sentence == tense(sentence, "past"):
      if sentence[ind_pos(sentence, "pronoun")][0] in {"i", "he", "she"}:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "was"))
      else:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "were"))
      return sentence
  else:
    sentence.insert(ind_pos(sentence, "aux"), ("be", "being"))
    return sentence
  pass

eq_list_1 = """v_0

f_past
 v_0

f_because
 f_past
  v_0
 f_join
  f_past
   v_1
  f_past
   v_2"""

eq_list_1 =  eq_list_1.split("\n\n")
def gen():
  list_basic = []
  for item in verb:
    list_basic.append([("pronoun", "he"), ("verb", item[1])])
    list_basic.append([("pronoun", "i"), ("verb", item[0])])
    list_basic.append([("pronoun", "he"), ("aux", "is"), ("verb", item[3])])
  for item in adjective:
    list_basic.append([("pronoun", "he"), ("aux", "is"), ("adjective", item)])
    list_basic.append([("pronoun", "i"), ("aux", "am"), ("adjective", item)])

  return list_basic

s_1 = list(zip("pronoun aux adjective".split(" "), "i am honest".split(" ")))
def generate_question(s_1):
#print(invert(s_1))
  extra_word =[("adverb", "away")]
  right_of = []
  for item in extra_word:
    if item in s_1:
      right_of.append([s_1[s_1.index(item)-1], item])
      s_1.remove(item)
  #print(inter(s_1))
  #s_1 = list(zip("pronoun aux verb".split(" "), "i killed".split(" ")))
  #s_2 = list(zip("pronoun aux verb".split(" "), "he was killed".split(" ")))
  #print(joins(s_1, s_2))
  question = []
  for eq in eq_list_1:
    if eq == "v_0":
      for fx in [inter, inter2]:
        
        question.append(" ".join(break_s(fx(s_1))[1]) + " -> " + " ".join(break_s(s_1)[1]))
    i = 0
    while "v_"+str(i) in eq:
      i += 1
    #print(i)
    for item in itertools.permutations(gen(), i):
      #print(" ".join(break_s(compute(list(item), tree_form(eq)))[1]))
      store = []
      tmp  = compute(list(item), tree_form(eq))
        
      if tmp == s_1:
        for x in  list(item)+store+[s_1]:
          for h in right_of:
            if h[0] in x:
              x.insert(x.index(h[0])+1, h[1])
          y=None
          if x == "can't simplify":
            continue
          tmp = tense(copy.deepcopy(x), "past")
          if tmp == "can't simplify":
            tmp = x
          for fx in [inter, inter2]:
            tmp_2 = fx(tmp)
            if tmp_2 != "can't simplify":
              y = tmp_2
            else:
              continue
            question.append(" ".join(break_s(y)[1]) + " -> " + " ".join(break_s(tmp)[1]))
          i=-1
        break
    if i == -1:
      break
  for item in list(set(question)):
    print(item)

generate_question(s_1)
print("above")
"""
for eq in eq_list_1:
  for item in gen():
    print(" ".join(break_s(compute([item], tree_form(eq)))[1]))
for eq in eq_list_2:
  for item in itertools.permutations(gen(), 2):
    print(" ".join(break_s(compute(list(item), tree_form(eq)))[1]))
for eq in eq_list_3:
  for item in itertools.permutations(gen(), 3):
    print(" ".join(break_s(compute(list(item), tree_form(eq)))[1]))
print()
"""

s_2 = list(zip("pronoun aux verb".split(" "), "he was killed".split(" ")))
s_3 = list(zip("pronoun aux verb".split(" "), "i am running".split(" ")))

s_b = joins(s_1, s_2)
s_c = because(s_3, s_b)
s_b = red_because(s_3, s_b)
print(" ".join(break_s(s_c)[1]))
print(" ".join(break_s(s_b)[1]))
print()

s_a = invert(inter(copy.deepcopy(s_1)))
print(" ".join(break_s(s_a)[1]))
print()

s_a = tense(copy.deepcopy(s_1), "past")
print(" ".join(break_s(s_a)[1]))
s_a = invert(s_a)
print(" ".join(break_s(s_a)[1]))
s_a = invert(s_1)
print(" ".join(break_s(s_a)[1]))
s_a = tense(copy.deepcopy(s_a), "past")
print(" ".join(break_s(s_a)[1]))

print()
print()

eq = """hi
#hi
##asl ?
###F
####Alex. M 24 here.*What is your name?
#####Sofia
######Hi Sofia
#######hi
#hey
##sup?
###okay u?
####same.
#####...*boring
######cute"""
import random
eq = eq.replace(" ", ">")
eq = eq.replace("#", " ")
eq_tree = tree_form(eq)
def gen_tree(tree):
  print(tree.name.replace(">", " ").replace("*", "\n"))
  if not tree.children:
    return
  choice = random.choice(tree.children)
  gen_tree(choice)
gen_tree(eq_tree)
