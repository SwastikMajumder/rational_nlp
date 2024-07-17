import copy
import itertools
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

# fancy print
def print_equation_helper(equation_tree):
    if equation_tree.name[:2] in {"v_", "s_"}:
        return equation_tree.name # leaf node
    s = equation_tree.name[2:] + "(" # bracket
    x =[]
        
    for child in equation_tree.children:
      x.append(print_equation_helper(child))
    s += ",".join(x)
    s += ")"
    return s
n_var = 0
# fancy print main function
def print_equation(eq):
    global n_var
    #eq = str_form(flatten_tree(tree_form(eq)))
    """
    if n_var == 0:
      eq = eq.replace("v_0", "x")
      eq = eq.replace("v_1", "y")
      eq = eq.replace("v_2", "z")
    elif n_var == 1:
      eq = eq.replace("v_0", "y")
      eq = eq.replace("v_1", "x")
      eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")
    """
    return print_equation_helper(tree_form(eq))

noun_single = ["human", "rock", "ball", "table"]
noun_plural = ["humans", "rocks", "balls", "tables"]
#noun_special = ["something"]
noun = noun_single + noun_plural
verb_single = [("push", "pushes", "pushing", "pushed", "pushed"), ("throw", "throws", "throwing", "threw", "thrown"), ("go", "goes", "going", "went", "gone"), ("run", "runs", "running", "ran", "run"), ("want", "wants", "wanting", "wanted", "wanted"), ("sleep", "sleeps", "sleeping", "slept", "slept"), ("laugh", "laughs", "laughing", "laughed", "laughed")]
verb_double = [("annoy", "annoys", "annoying", "annoyed", "annoyed"), ("kill", "kills", "killing", "killed", "killed"), ("scare", "scares", "scaring", "scared", "scared")]
verb = verb_single + verb_double
proper_he = ["john"]
proper_she = ["mary"]
proper_the_it = ["wall"]
proper = proper_he + proper_she + proper_the_it
adjective = ["happy", "honest", "genius", "fine", "good", "fascinating", "silent", "mediocre"]

def ind_pos_2(sentence, pos, ind=0):
  for i, item in enumerate(sentence):
    if item[ind] == pos:
      return i
  return -1

def ind_pos(sentence, pos, ind=0):
  for i, item in enumerate(sentence):
    if item[ind] == pos or (ind == 0 and pos=="pronoun" and item[ind] == "proper"):
      return i
  return -1

pronoun_list = [("he", "him"), ("i", "me"), ("she", "her"), ("they", "their"), ("this", "this"), ("it", "it")]

def because(a, b):
  if a[-1][0] == "aux":
    return "can't simplify"
  if len(a) == 1 or len(b) == 1:
    return "can't simplify"
  return a + [("conjunction", "because")] + b

def red_because(a, b):
  word = None
  global pronoun_list
  if b[0][0] == "pronoun":
    for item in pronoun_list:
      if item[0]==b[0][1]:
        word = item[1]
        
  return a + [("conjunction", "because"), ("preposition", "of"), ("pronoun", word)]

def prep_of(sentence, of_whom):
  sentence = copy.deepcopy(sentence)
  of_whom = copy.deepcopy(of_whom)
  if len(of_whom) != 2:
    return "can't simplify"
  if len(sentence) != 3:
    return "can't simplify"
  if replace_tense_word(sentence, 0)[2] != replace_tense_word(of_whom, 0)[1]:
    return "can't simplify"
  verb_of = ["scared"]
  dic_pronoun = {"i": "me", "he": "him", "they": "them"}
  if sentence[0][1] == of_whom[0][1]:
    return "can't simplify"
  for item in verb_of:
    if ind_pos(sentence, item, 1) != -1:
      sentence.insert(ind_pos(sentence, item, 1)+1, ("preposition", "of"))
      return sentence + [("pronoun", dic_pronoun[of_whom[0][1]])]

  return "can't simplify"
def perfect(sentence):
  return replace_tense_word(sentence, 4)
def joins(a, b):
  a = copy.deepcopy(a)
  b = copy.deepcopy(b)
  #print(a, b)
  if b[-1][1] in {"at", "something", "to"}:
    b.pop(-1)
  if ind_pos(b, "aux") == -1:
    return "can't simplify"
  if ind_pos(a, "adjective") != -1 or ind_pos(b, "adjective") != -1:
    return "can't simplify"
  
  if replace_tense_word([a[ind_pos(a, "verb")]], 3) != replace_tense_word([b[ind_pos(b, "verb")]], 3) or b[-1][0] == "aux":
    return "can't simplify"
  #print("HHI")
  word = None
  if b[0][0] in {"pronoun"}:
    for item in pronoun_list:
      if item[0]==b[0][1]:
        word = item[1]
    output = a + [("pronoun", word)]
  elif b[0][0] == "proper":
    output = a + [b[0]]
  #print(output)
  
  if len(output) not in {3, 5} or (len(output)==4 and ind_pos(output, "determiner") == -1):
    return "can't simplify"
  #print("HII")
  if a[0][1] == b[0][1]:
    return "can't simplify"
  verb_at = ["laughed", "rock"]
  verb_to = ["ball", "table"]
  for item in verb_at:
    if ind_pos(output, item, 1) != -1:
      output.insert(ind_pos(output, item, 1)+1, ("preposition", "at"))
      return output
  for item in verb_to:
    if ind_pos(output, item, 1) != -1:
      output.insert(ind_pos(output, item, 1)+1, ("preposition", "to"))
      return output
  word = replace_tense_word(a, 0)[ind_pos(replace_tense_word(a, 0), "verb")]
  #print(output)
  for item in verb_double:
    if item[0] == word[1]:
      return output
  return "can't simplify"
  #return output

#def and_sentence(sentence):
#  pass

def first_person(sentence):
  sentence = copy.deepcopy(sentence)
  if ind_pos(sentence, "is", 1) != -1:
    sentence[ind_pos(sentence, "is", 1)] = ("aux", "am")
  if ind_pos(sentence, "are", 1) != -1:
    sentence[ind_pos(sentence, "are", 1)] = ("aux", "am")
  if ind_pos(sentence, "you", 1) != -1:
    sentence[ind_pos(sentence, "you", 1)] = ("pronoun", "i")
  return sentence
def inter2(sentence):
  output = []
  if len(sentence) == 1:
    return "can't simplify"
  if len(sentence) == 2 and ind_pos(sentence, "determiner") != -1:
    return "can't simplify"
  if ind_pos(sentence, "something", 1) != -1:
    return "can't simplify"
  if ind_pos(sentence, "because", 1) != -1:
    return "can't simplify"
    #a = copy.deepcopy(sentence[:ind_pos(sentence, "because", 1)])
    #print(a)
    #return inter(a) + [("preposition", "after")] + sentence[ind_pos(sentence, "because", 1)+1:]
  if sentence[0][0] in {"pronoun", "proper"} or sentence[0][0] == "noun":
    if sentence[0][1] == "this":
      output.append(("question", "what"))
    else:
      output.append(("question", "who"))
  if ind_pos(sentence, "aux") != -1:
    if sentence[ind_pos(sentence, "aux")] != ("aux", "am"):
      output.append(sentence[ind_pos(sentence, "aux")])
    else:
      output.append(("aux", "is"))
  if ind_pos(sentence, "verb") != -1:
    output.append(sentence[ind_pos(sentence, "verb")])
  if ind_pos(sentence, "preposition") != -1:
    output.append(sentence[ind_pos(sentence, "preposition")])
  if sentence[-1] == ("pronoun", "him"):
    output.append(sentence[-1])
  if ind_pos(sentence,"determiner") != -1 and ind_pos(sentence,"noun") != -1:
    output.append(sentence[ind_pos(sentence, "determiner")])
    output.append(sentence[ind_pos(sentence, "noun")])
    return output
  if ind_pos(sentence, "adjective") != -1:
    output.append(sentence[ind_pos(sentence, "adjective")])
  if ind_pos(sentence, "adverb") != -1:
    output.append(sentence[ind_pos(sentence, "adverb")])
  
  return output

def invert(sentence_orig):
  sentence = copy.deepcopy(sentence_orig)
  if len(sentence_orig) == 1:
    return "can't simplify"
  if "not" in " ".join(break_s(sentence)[1]):
    return "can't simplify"
  index = ind_pos(sentence, "adjective")
  if index == -1:
    index = ind_pos(sentence, "verb")
  else:
    sentence.insert(index, ("adverb", "not"))
    return sentence
  if index == -1:
    if ind_pos(sentence, "question") != -1:
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
  if len(sentence) == 1:
    return "can't simplify"
  output = []
  
  #if sentence == replace_tense_word(sentence, 2):
    
  if ind_pos(sentence, "preposition") != -1:
    return "can't simplify"
  #sentence = copy.deepcopy(sentence)
  #sentence.remove(("adverb", "away"))
  if ind_pos(sentence, "because", 1) != -1:
    output.append(("question", "why"))
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
    if ind_pos(sentence, "verb") != -1:
      output.append(sentence[ind_pos(sentence, "aux")])
      output.append(sentence[ind_pos(sentence, "pronoun")])
      output.append(sentence[ind_pos(sentence, "verb")])
      return output
  if ind_pos(sentence, "adjective") != -1:
    output.append(("question", "how"))
  if ind_pos(sentence, "verb") != -1:
    output.append(("question", "what"))
  if ind_pos(sentence, "verb") != -1:
    if sentence[1][0] == "aux":
      if sentence == replace_tense_word(sentence, 2):
        output.append(sentence[1])
        output.append(sentence[ind_pos(sentence, "pronoun")])
        output.append(("do", "doing"))
        return output
      #print()
      #if sentence[-1][1] not in [item[4] for item in verb_double]:
      #  return "can't simplify"
      output += [("verb", "happened"), ("preposition", "to")]
      for item in pronoun_list:
        if item[0] == sentence[0][1]:
          output.append(("pronoun", item[1]))
      return output
    if sentence == tense(sentence, "past"):
      output.append(("do", "did"))
      if ind_pos(sentence, "noun") != -1:
        output.append(sentence[ind_pos(sentence, "pronoun")])
        output.append(sentence[ind_pos(sentence, "verb")])
        return output
    else:
      if sentence[0] == ("pronoun", "i"):
        output.append(("do", "do"))
      else:
        output.append(("do", "does"))
    output.append(sentence[ind_pos(sentence, "pronoun")])
    output.append(("do", "do"))
  else:
    if ind_pos(sentence, "noun") != -1:
      if sentence[0][1] in {"a", "an"}:
        word = sentence[1][1]
        if word in noun_single:
          word = noun_plural[noun_single.index(word)]
        return [("question", "how"), ("adjective", "many")] + [("noun", word)]
      output.append(("question", "who"))
    output.append(sentence[ind_pos(sentence, "aux")])
    if ind_pos(sentence, "pronoun") != -1:
      output.append(sentence[ind_pos(sentence, "pronoun")])
    return output
  if ind_pos(sentence,"determiner") != -1:
    output.append(sentence[ind_pos(sentence, "determiner")])
  if ind_pos(sentence,"noun") != -1:
    output.append(sentence[ind_pos(sentence, "noun")])
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

def by_sentence(a, b):
  word = None
  if a[0][0] == "proper":
    if a[0][1] in proper_he:
      word = "he"
    else:
      word = "she"
  elif a[0][0] == "pronoun":
    word = a[0][1]
  if word is None or (b[0][1] != word and b[0] != a[0]):
    return "can't simplify"
  
  return a + [("conjunction", "by")] + replace_tense_word(b, 2)[1:]
  pass

def tense(sentence, num):
  global verb
  sto = []
  if sentence[-1][1] == "at":
    sto = [("preposition", "at")]
  if len(sentence) == 1:
    return "can't simplify"
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
def compute(param, param2, equation):
  global store
  output = []
  
  equation = copy.deepcopy(equation)
  for item in equation.children:
    if item.name[:2] == "v_":
      output.append(param[int(item.name[2:])])
    elif item.name[:2] == "s_":
      output.append(param2[int(item.name[2:])])
    else:
      tmp = compute(param, param2, item)
      if tmp == "can't simplify":
        return tmp
      else:
        #print(" ".join(break_s(tmp)[1]))
        store.append(tmp)
        output.append(tmp)
    
  output = copy.deepcopy(output)
  ans = None
  if equation.name == "f_invert":
    ans = invert(*output)
  elif equation.name == "f_cont":
    ans = cont(*output)
  elif equation.name == "f_past":
    ans = tense(*output, "past")
  elif equation.name == "f_by":
    ans = by_sentence(*output)
  elif equation.name == "f_join":
    ans = joins(*output)
    #print(ans)
  elif equation.name == "f_joina":
    ans = joins2(*output)
  elif equation.name == "f_because":
    ans = because(*output)
  elif equation.name == "f_rbecause":
    ans = red_because(*output)
  elif equation.name == "f_of":
    ans = prep_of(*output)
  elif equation.name == "f_word":
    ans = add_word(*output)
  elif equation.name == "f_proper":
    ans = proper_noun(*output)
  elif equation.name == "f_perfect":
    ans = perfect(*output)
  #print(ans)
  return copy.deepcopy(ans)

def add_word(a, b):
  if b[0][0] == "proper":
    return "can't simplify"
  #print("hi")
  #if ind_pos(a, "verb") != -1 and ind_pos(a, "aux") != -1 and b[0][0] == "adverb":
  #  return "can't simplify"
  
  if ind_pos(a, "adjective") != -1 or len(a) == 1 or (len(a) == 2 and a[1][0] not in {"verb", "aux"}) or len(b) > 2 or (len(b) == 2 and b[1][0] in {"verb", "aux"}):
    return "can't simplify"
  
  #d = {"away": "adverb"}
  #sentence = copy.deepcopy(sentence)
  #output= sentence + [(d[word], word)]
  #print("start")
  #print(sentence)
  #print(output)
  #print("end")
  return a+b

def semantic(sentence, equation, param):
  if sentence[-1][0] == "aux":
    #print("HIHI")
    return "can't simplify"
  if tree_form(equation).name == "f_because":
    left = str_form(tree_form(equation).children[0])
    right = str_form(tree_form(equation).children[1])
    if ("f_invert" in left and "f_invert" not in right) or ("f_invert" not in left and "f_invert" in right):
      return sentence
    left_v = []
    right_v = []
    for i in range(len(param)):
      if "v_" + str(i) in left:
        left_v.append(i)
      if "v_" + str(i) in right:
        right_v.append(i)
    #print(param[left_v[0]], param[right_v[0]])
    def examine(sentence):
      sentence = copy.deepcopy(sentence)
      if ind_pos(sentence, "aux") != -1:
        sentence.pop(ind_pos(sentence, "aux"))
      return replace_tense_word(sentence, 0)
    if len(left_v) == len(right_v) and all(examine(param[left_v[i]]) == examine(param[right_v[i]]) for i in range(len(left_v))):
      return "can't simplify"
  return sentence
  

def joins2(a, b):
  if len(a) == 1 or len(b) == 1:
    return "can't simplify"
  word = None
  a = copy.deepcopy(a)
  #print(a)
  for item in pronoun_list:
    if item[0] == b[0][1]:
      word = item[1]
  return a + [("preposition", "of")] + [("pronoun", word)]
def proper_noun(sentence, word):
  sentence = copy.deepcopy(sentence)
  word = copy.deepcopy(word)
  if len(word) != 1:
    return "can't simplify"
  if ind_pos(sentence, "pronoun") != -1:
    #print(word[0][1])
    if sentence[ind_pos(sentence, "pronoun")][1] == "it" and word[0][1] in proper_the_it:
      sentence[ind_pos(sentence, "pronoun")] = word[0]
      sentence.insert(ind_pos(sentence, "pronoun"), ("determiner", "the"))
      return sentence
    if (sentence[ind_pos(sentence, "pronoun")][1] in {"he", "him"} and word[0][1] in proper_he) or\
       (sentence[ind_pos(sentence, "pronoun")][1] in {"her", "she"} and word[0][1] in proper_she):
      #print("JHIHI")
      sentence[ind_pos(sentence, "pronoun")] = word[0]
      #print(sentence)
      return sentence
  return "can't simplify"
def cont(sentence):
  if ind_pos(sentence, "noun") != -1 or len(sentence) == 1:
    return "can't simplify"
  sentence = copy.deepcopy(sentence)
  if ind_pos(sentence, "adjective") != -1:
    return sentence
  if ind_pos(sentence, "aux") == -1:
    if sentence == tense(sentence, "past"):
      if sentence[ind_pos(sentence, "pronoun")][1] in {"they", "you"}:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "were"))
      else:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "was"))
      sentence = replace_tense_word(sentence, 2)
      return sentence
    else:
      if sentence[ind_pos(sentence, "pronoun")][1] in {"i"}:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "am"))
      elif sentence[ind_pos(sentence, "pronoun")][1] in {"they", "you"}:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "are"))
      else:
        sentence.insert(ind_pos(sentence, "verb"), ("aux", "is"))
      sentence = replace_tense_word(sentence, 2)
      return sentence
  else:
    #if tense(sentence, "past") == sentence:
    #  sentence.insert(ind_pos(sentence, "verb"), ("aux", "was"))
    #else:
    #  sentence.insert(ind_pos(sentence, "verb"), ("aux", "is"))
    if sentence == replace_tense_word(sentence, 2):
      return "can't simplify"
    else:
      replace_tense_word(sentence, 2)
    return sentence
  pass



#print(print_equation(eq_list_1))
eq_list_1 = """f_join
 f_word
  f_proper
   f_past
    v_0
   s_0
  s_1
 f_proper
  v_1
  s_2"""

eq_list_1 = """f_join
 f_proper
  f_past
   v_0
  s_0
 f_proper
  f_past
   v_1
  s_1"""

eq_list_1 = """f_by
 f_join
  f_proper
   f_past
    v_0
   s_0
  f_proper
   f_past
    v_1
   s_1
 f_join
  f_word
   f_past
    v_2
   s_2
  v_3"""

eq_list_1= """f_join
 f_word
  f_past
   v_0
  s_0
 v_1"""

eq_list_1 =  eq_list_1.split("\n\n")
adverb = ["away", "fast"]

part_of_speech = {"not": "adverb"}
for item in verb:
  for s in item:
    part_of_speech[s] = "verb"
for item in adjective:
  part_of_speech[item] = "adjective"
for item in ["is", "am", "are", "is", "was", "were"]:
  part_of_speech[item] = "aux"
for item in ["i", "he", "her", "this", "him", "she", "they", "me", "you", "it"]:
  part_of_speech[item] = "pronoun"
for item in ["who", "how", "what", "why", "when", "where"]:
  part_of_speech[item] = "question"
for item in adverb:
  part_of_speech[item] = "adverb"
for item in ["because", "by"]:
  part_of_speech[item] = "conjunction"
for item in ["of", "at", "to"]:
  part_of_speech[item] = "preposition"
for item in noun + ["something"]:
  part_of_speech[item] = "noun"
for item in proper:
  part_of_speech[item] = "proper"
part_of_speech["a"] = "determiner"
part_of_speech["an"] = "determiner"
part_of_speech["the"] = "determiner"
for item in ["do", "doing", "does", "did"]:
  part_of_speech[item] = "do"
"""
s = "he throws, john, a ball, she is thrown to, mary".split(", ")
#print(s)
for i in range(len(s)):
  sc = s[i]
  sc = sc.split(" ")
  sd = [part_of_speech[item] for item in sc]
  s_1 = list(zip(sd, sc))
  s[i] = s_1
"""
#for i in range(1):
sc = input("input your sentence: ")
sc = sc.split(" ")
sd = [part_of_speech[item] for item in sc]
s_1 = list(zip(sd, sc))
if ind_pos(s_1, "question") != -1:
  s_1 = first_person(s_1)
#s.append(s_1)
  
#print(by_sentence(joins(tense(s[0],"past"),tense(s[1],"past")),joins(add_word(tense(s[2],"past"),s[3]),s[4])))
#print(by_sentence(joins(proper_noun(tense(s[0],"past"), s[1]),proper_noun(tense(s[2],"past"),s[3])),joins(add_word(tense(s[4],"past"),s[5]),s[6])))
#print(joins(add_word(proper_noun(tense(s[0],"past"),s[1]),s[2]),proper_noun(s[3],s[4])))

#s_1 = s[0]

#print(by_sentence(s_2, s_1))
#print(tense(s_1, "past"))
#print(s_1)
#print(add_word(cont(s_1), [("adverb", "away")]))
#print(inter(cont(s_1)))
#print(inter(tense(s_1, "past")))
def apply_fx(pos, sentence, question_index):
  if question_index != 1:
    return pos
  elif not isinstance(pos[0], tuple):
    return [pos[0]]
  if ind_pos(sentence, "verb") != -1:
    word = replace_tense_word([sentence[ind_pos(sentence, "verb")]], 0)[0][1]
    #print(word)
    for item in verb:
      if item[0] == word:
        return [item]
  return [pos[0]]
def gen2():
  list_basic = []
  
  for item in noun_single:
    list_basic.append([("determiner", "the"), ("noun", item)])
    if item[0] in {"a", "e", "i", "o", "u"}:
      list_basic.append([("determiner", "an"), ("noun", item)])
      list_basic.append([("determiner", "an"), ("noun", item)])
    else:
      list_basic.append([("determiner", "a"), ("noun", item)])
      list_basic.append([("determiner", "a"), ("noun", item)])

  for item in adverb:
    list_basic.append([("adverb", item)])
  for item in proper:
    list_basic.append([("proper", item)])
  list_basic.append([("preposition", "at")])
  final = []
  
  list_basic = [list(item) for item in list(set([tuple(item) for item in list_basic]))]

  for sen in list_basic:
    if any(item[0] in {"adjective", "adverb", "proper", "preposition"} and item[1] not in sc for item in sen):
      final.append(sen)
    if any(item[0] == "noun" and item[1]!="something" and item[1] not in sc for item in sen):
      final.append(sen)
      
  for item in final:
    if item in list_basic:
      list_basic.remove(item)
  
  #for item in list_basic:
  #  print(break_s(item)[1])
  #uhidcuh()
  return list_basic
def gen():
  list_basic = []
  #question_index = 1
  
  list_basic.append([("pronoun", "he"), ("aux", "is")])
  list_basic.append([("pronoun", "i"), ("aux", "am")])
  for item in verb:
    list_basic.append([("pronoun", "she"), ("verb", item[1])])
    list_basic.append([("pronoun", "he"), ("verb", item[1])])
    list_basic.append([("pronoun", "i"), ("verb", item[0])])
    list_basic.append([("pronoun", "he"), ("aux", "is"), ("verb", item[4])])
    list_basic.append([("pronoun", "she"), ("aux", "is"), ("verb", item[4])])
    list_basic.append([("pronoun", "it"), ("aux", "is"), ("verb", item[4])])
    list_basic.append([("pronoun", "i"), ("aux", "am"), ("verb", item[4])])
  for item in adjective:
    if item not in sc:
      continue
    list_basic.append([("pronoun", "he"), ("aux", "is"), ("adjective", item)])
    list_basic.append([("pronoun", "i"), ("aux", "am"), ("adjective", item)])
  
  final = []
  p = []
  v = []
  list_basic = [list(item) for item in list(set([tuple(item) for item in list_basic]))]
  for i in range(len(sc)):
    if sd[i] in {"pronoun"}:
      for item in pronoun_list:
        if sc[i] == item[0] or sc[i] == item[1]:
          p.append(item[0])
          p.append(item[1])
    elif sd[i] == "verb":
      v.append(replace_tense_word([("verb", sc[i])], 0)[0][1])

  for sen in list_basic:
    if any(item[0]=="verb" and replace_tense_word([item], 0)[0][1] not in v for item in sen):
      #print(sen)
      final.append(sen)
      
  for item in final:
    if item in list_basic:
      list_basic.remove(item)
  
  #for item in list_basic:
  #  print(break_s(item)[1])
  #uhidcuh()
  return list_basic

#s_1 = list(zip("pronoun aux verb adverb".split(" "), "i am running away".split(" ")))

#print(inter(tense(add_word(cont(s_1), "away"),"past")))
#s_2 = list(zip("pronoun aux verb".split(" "), "i am scared".split(" ")))
#print(joins2(s_2, joins(s_1, s_2)))
#ufcurf()
#print(cont(s_1))
def generate_question(s_1):
#print(invert(s_1))
  #print(inter(s_1))
  #s_1 = list(zip("pronoun aux verb".split(" "), "i killed".split(" ")))
  #s_2 = list(zip("pronoun aux verb".split(" "), "he was killed".split(" ")))
  #print(joins(s_1, s_2))
  global store
  eq_store = []
  question = []
  for eq in eq_list_1:
    #print(eq)
    if eq == "v_0":
      for item in gen():
        for fx in [inter, inter2]:
          if fx(item) != "can't simplify" and fx(item) == s_1:
            if question_input == 1:
              eq_store.append(eq)
              break
            question.append(" ".join(break_s(item)[1]))
        if eq in eq_store:
          break

    i = 0
    while "v_"+str(i) in eq:
      i += 1
    j = 0
    while "s_"+str(j) in eq:
      j += 1
      
    if eq == "v_0":
      continue
    
    for item2 in itertools.permutations(gen2(), j):
      for item in itertools.permutations(gen(), i):
        #print(list(item)+list(item2))
        store = []
        tmp  = semantic(compute(list(item), list(item2), tree_form(eq)), eq, list(item)+list(item2))
        #tmp  = compute(list(item), tree_form(eq))
        
        #if tmp != "can't simplify":
        #  print(" ".join(break_s(tmp)[1]))
        #continue
        if tmp == s_1:
          print(item)
          print(item2)
          print()
          for x in  list(item)+store+[s_1]:
            y=None
            if x == "can't simplify":
              continue
            tmp = x
            if tmp == "can't simplify":
              tmp = x
            for fx in [inter, inter2]:
              tmp_2 = fx(tmp)
              if tmp_2 != "can't simplify":
                y = tmp_2
              else:
                continue
              question.append(" ".join(break_s(y)[1]) + " -> " + " ".join(break_s(tmp)[1]))
          

  return list(set(question))

q=generate_question(s_1)
print()
for item in list(set(q)):
  print(item)
