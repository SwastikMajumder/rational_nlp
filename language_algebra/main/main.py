import copy

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
    def does_given_equation_satisfy_formula_lhs_structure(equation, formula_lhs):
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
            if does_given_equation_satisfy_formula_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    if structure_satisfy:
      return does_given_equation_satisfy_formula_lhs_structure(equation, formula_lhs)
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
            if does_given_equation_satisfy_formula_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: # if formula lhs structure is satisfied by the equation given
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
import itertools
log = []
def log_process(file_name):
    global log
    
    x, y = return_formula_file(file_name) # load formula file
    output = []
    for i in range(len(x)):
        
        input_f = x[i]
        output_f = y[i]
        if input_f.name != "f_condition":
            continue
        
        
        a = len(input_f.children[0].children)
        b = len(input_f.children[1].children)
        log_1 = []
        for item in log:
            #print(string_equation(item))
            for sitem in input_f.children[0].children:
                #print(string_equation(str_form(sitem)))
                if apply_individual_formula_on_given_equation(tree_form(item), sitem, None, False, True):
                    log_1.append(item)
        #print()
        for item in log_1:
            print(string_equation(item))
        for present in itertools.permutations(list(set(log_1)), r=a):
            present = [tree_form(item) for item in present]
            eq = TreeNode("f_present", present)
            tmp = apply_individual_formula_on_given_equation(eq, input_f.children[0], output_f)
            
            if tmp != []:
                
                if b == 0:
                    
                    output += tmp
                else:
                    if all(apply_individual_formula_on_given_equation(\
                        TreeNode("f_condition", [TreeNode("f_present", present), TreeNode("f_notpresent", [tree_form(notpresent)])])\
                                                                      , input_f, output_f) == []\
                        for notpresent in log):
                        output += tmp
    #if all(item in log for item in output)
    return list(set(output))

# Function to generate neighbor equations
def generate_transformation(equation, file_name):
    input_f, output_f = return_formula_file(file_name) # load formula file
    transformed_equation_list = []
    for i in range(len(input_f)): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to generate neighbor equations
def generate_arithmetical_transformation(equation):
    transformed_equation_list = []
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) # perform arithmetic
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to read formula file
def return_formula_file(file_name):
    with open(file_name, 'r') as file:
      content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f] # return


import syntax
def search(equation, depth, file_list, auto_arithmetic=True, visited=None):
    global log
    if depth == 0: # limit the search
        return None
    if visited is None:
        visited = set()
    if equation in visited:
        return None
    visited.add(equation)
    print(string_equation(equation))
    #if tree_form(equation).name == "f_S":
    #    print(syntax_3_4.process_2(str_form(tree_form(equation).children[0])))
    output =[]
    if file_list[0]:
      output += generate_transformation(equation, file_list[0])
    #if auto_arithmetic:
    #  output += generate_arithmetical_transformation(equation)
    if len(output) > 0:
      output = [output[0]]
    else:
      if file_list[1]:
        output += generate_transformation(equation, file_list[1])
      #if not auto_arithmetic:
      #  output += generate_arithmetical_transformation(equation)
      if file_list[2] and len(output) == 0:
          output += generate_transformation(equation, file_list[2])
    output = [item for item in output if item.count("f_instrument") <= 1]
    
    for i in range(len(output)):
        result = search(output[i], depth-1, file_list, auto_arithmetic, visited) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    output = list(set(output))
    return output

# fancy print
def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    #if len(equation_tree.children) == 1:
    s = equation_tree.name[2:] + s
    sign = {"f_add": "+", "f_mul": "*", "f_pow": "^", "f_div": "/", "f_int": ",", "f_sub": "-", "f_dif": "?", "f_sin": "?", "f_cos": "?", "f_tan": "?", "f_eq": "=", "f_sqt": "?"} # operation symbols
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + ","
    s = s[:-1] + ")"
    return s

# fancy print main function
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

def break_equation(equation):
    sub_equation_list = [equation]
    equation = tree_form(equation)
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(str_form(child)) # collect broken equations
    return sub_equation_list

def is_child_of(equation, fx_name):
    return any(tree_form(part).name[2:] == fx_name and "token_found" in part for part in break_equation(equation))

def remove_past(equation):
    if equation.name in {"f_cont", "f_past"}:
        return equation.children[0]
    coll = TreeNode(equation.name, [])
    for child in equation.children:
        coll.children.append(remove_past(child))
    return coll

eq = """f_S
 f_past
  f_verb3
   d_john
   d_kill
   d_mary"""

eq_2 = """f_S
 f_that
  f_verb2
   d_the-police
   d_know
  f_past
   f_verb3
    d_john
    d_kill
    d_mary"""
log += [eq]
output = [eq]
#output = [eq, eq_2]
#log += [eq, eq_2]
for item in output:
    log += search(item, 5, [None, "simple2_law.txt", None])
for _ in range(2):
    tmp = log_process("simple2_law.txt")
    log += tmp
    if tmp == []:
        break
    for item in tmp:
        log += search(item, 5, [None, "simple2_law.txt", None])
log = list(set(log))
print()
for item in log:
    print(syntax.process_2(str_form(tree_form(item).children[0])))
