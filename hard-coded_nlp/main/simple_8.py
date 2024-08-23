import copy
import vocabulary 
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
            if all(node_type(item.name) == "d_" for item in equation.children): # if only numbers
                x = []
                for item in equation.children:
                    x.append(item) # convert string into a number
                if equation.name == "f_eq":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        if x[0].name == x[1].name:
                            return tree_form("d_true")
                        else:
                            return tree_form("d_false")
                if equation.name[:3] == "f_w":
                    count_target_node -= 1
                    if count_target_node == 0:
                        data = vocabulary.vocab[equation.name[2:]]
                        if x[0].name[2:] in data.keys():
                            return tree_form("d_" + data[x[0].name[2:]])
                        else:
                            return tree_form("d_" + x[0].name[2:])
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


# Function to generate neighbor equations
def generate_transformation(equation, file_name):
    input_f, output_f = return_formula_file(file_name) # load formula file
    transformed_equation_list = []
    for i in input_f.keys(): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to generate neighbor equations
def generate_transformation_2(equation, file_name):
    input_f, output_f = return_formula_file(file_name) # load formula file
    transformed_equation_list = []
    for i in range(len(input_f.keys())): # go through all formulas and collect if they can possibly transform
        tmp = apply_individual_formula_on_given_equation(tree_form(equation), copy.deepcopy(input_f[str(i)]), copy.deepcopy(output_f[str(i)]))
        if tmp != []:
            #print(tmp)
            return tmp
    return []

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
    input_f ={}
    output_f = {}
    for i in range(0, len(x), 2):
        input_f[str(int(i/2))] = tree_form(x[i])
    for i in range(1, len(x), 2):
        output_f[str(int((i-1)/2))] = tree_form(x[i])
    return [input_f, output_f] # return

log = None

def search(equation, depth, file_list, auto_arithmetic=True, visited=None):
    global log
    if depth == 0: # limit the search
        return None
    if visited is None:
        visited = set()
    if equation in visited:
        return None
    visited.add(equation)
    #print(string_equation(equation))
    def extract(eq):
      output = []
      def helper(eq):
        if eq.name[:2] == "d_":
          return output.append(eq.name[2:])
        for child in eq.children:
          helper(child)
      helper(eq)
      return output
    def syntax(eq):
        return " ".join(extract(tree_form(eq))).replace("-", " ").replace(" # ", "")
    if "f_" not in equation.replace("f_sentence", "g_sentence"):
        log = syntax(equation)
    #if tree_form(equation).name == "f_S":
    #    print(syntax_3_4.process_2(str_form(tree_form(equation).children[0])))
    output = []
    if file_list[0]:
      output = generate_transformation_2(equation, file_list[0])
    if output == [] and auto_arithmetic:
      output = generate_arithmetical_transformation(equation)
    #print(output)
    if output == []:
      if file_list[1]:
        output += generate_transformation(equation, file_list[1])
      #if not auto_arithmetic:
      #  output += generate_arithmetical_transformation(equation)
      if file_list[2] and len(output) == 0:
          output += generate_transformation(equation, file_list[2])
    #print(output)
    
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

#output = [eq, eq_2]
#log += [eq, eq_2]
def process(eq):
    global log
    output = []
    for item in eq:
        #item = str_form(TreeNode("f_sentence", [tree_form(item)]))
        search(item, 10, ["simple3_law.txt", None, None])
        print(string_equation(item))
        output.append(log)
    print()
    for item in output:
        print(item)
    return output
"""
for _ in range(2):
    tmp = log_process("formula-list-5/simple2_law.txt")
    log += tmp
    if tmp == []:
        break
    for item in tmp:
        log += search(item, 5, [None, "formula-list-5/simple2_law.txt", None])
log = list(set(log))
print()
"""
