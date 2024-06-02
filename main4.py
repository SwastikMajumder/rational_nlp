import copy
import itertools

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

def print_equation_helper(equation_tree):
    if equation_tree.name[:2] in {"v_", "s_"}:
        return equation_tree.name
    s = equation_tree.name[2:] + "("
    x =[]
        
    for child in equation_tree.children:
      x.append(print_equation_helper(child))
    s += ",".join(x)
    s += ")"
    return s
def print_equation(eq):
    return print_equation_helper(tree_form(eq))

places = ["new_york"]
noun_single = ["human", "rock", "ball", "table"]
noun_plural = ["humans", "rocks", "balls", "tables"]

noun = noun_single + noun_plural
verb_single = [("push", "pushes", "pushing", "pushed", "pushed"), ("throw", "throws", "throwing", "threw", "thrown"), ("go", "goes", "going", "went", "gone"), ("run", "runs", "running", "ran", "run"), ("want", "wants", "wanting", "wanted", "wanted"), ("sleep", "sleeps", "sleeping", "slept", "slept"), ("laugh", "laughs", "laughing", "laughed", "laughed")]
verb_double = [("annoy", "annoys", "annoying", "annoyed", "annoyed"), ("kill", "kills", "killing", "killed", "killed"), ("scare", "scares", "scaring", "scared", "scared")]
verb = verb_single + verb_double
proper_he = ["john"]
proper_she = ["mary"]
proper_the_it = ["wall"]
proper_place = ["new_york"]
proper = proper_he + proper_she + proper_the_it + proper_place
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
  if b[-1][1] in {"at", "something", "to"}:
    b.pop(-1)
  if ind_pos(b, "aux") == -1:
    return "can't simplify"
  if ind_pos(a, "adjective") != -1 or ind_pos(b, "adjective") != -1:
    return "can't simplify"
  
  if replace_tense_word([a[ind_pos(a, "verb")]], 3) != replace_tense_word([b[ind_pos(b, "verb")]], 3) or b[-1][0] == "aux":
    return "can't simplify"
  word = None
  if b[0][0] in {"pronoun"}:
    for item in pronoun_list:
      if item[0]==b[0][1]:
        word = item[1]
    output = a + [("pronoun", word)]
  elif b[0][0] == "proper":
    output = a + [b[0]]
  elif b[0][0] == "determiner" and b[1][0] == "proper":
    output = a + b[:2]
  if len(output) not in {3, 5, 6} or (len(output)==4 and ind_pos(output, "determiner") == -1):
    return "can't simplify"
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
  for item in verb_double:
    if item[0] == word[1]:
      return output
  return "can't simplify"
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
    if sentence == tense(copy.deepcopy(sentence), "past"):
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
  if ind_pos(sentence, "preposition") != -1:
    return "can't simplify"
  if ind_pos(sentence, "because", 1) != -1:
    output.append(("question", "why"))
    es = sentence[:ind_pos(sentence, "because", 1)+1]
    if es == tense(es, "past"):
      if ind_pos(es, "verb") != -1 and es != replace_tense_word(es, 2):
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
  if ind_pos(sentence, "because", 1) != -1:
    return "can't simplify"
  if (num=="past" and ind_pos(sentence, "adjective") != -1) or (sentence == replace_tense_word(sentence, 2) and num=="past"):
    if ("aux", "is") in sentence:
      sentence[sentence.index(("aux", "is"))] = ("aux", "was")
    if ("aux", "am") in sentence:
      sentence[sentence.index(("aux", "am"))] = ("aux", "was")
    if ("aux", "are") in sentence:
      sentence[sentence.index(("aux", "are"))] = ("aux", "were")
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
  return sentence

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
    
  return copy.deepcopy(ans)

def add_word(a, b):
  if b[-1][1] in places:
    return a+ [("preposition","to")] + b
  if b[0][0] == "proper":
    return "can't simplify"
  if ind_pos(a, "adjective") != -1 or len(a) == 1 or (len(a) == 2 and a[1][0] not in {"verb", "aux"}) or len(b) > 2 or (len(b) == 2 and b[1][0] in {"verb", "aux"}):
    return "can't simplify"  
  return a+b

def semantic(sentence, equation, param):
  if sentence[-1][0] == "aux":
      
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
    if sentence[ind_pos(sentence, "pronoun")][1] == "it" and word[0][1] in proper_the_it:
      sentence[ind_pos(sentence, "pronoun")] = word[0]
      sentence.insert(ind_pos(sentence, "pronoun"), ("determiner", "the"))
      return sentence
    if (sentence[ind_pos(sentence, "pronoun")][1] in {"he", "him"} and word[0][1] in proper_he) or\
       (sentence[ind_pos(sentence, "pronoun")][1] in {"her", "she"} and word[0][1] in proper_she):
      sentence[ind_pos(sentence, "pronoun")] = word[0]
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
    if sentence == replace_tense_word(sentence, 2):
      return "can't simplify"
    else:
      replace_tense_word(sentence, 2)
    return sentence

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



eq_list_1= """f_join
 f_word
  f_past
   v_0
  s_0
 f_proper
  v_1
  s_1"""


eq_list_1= """f_proper
 v_0
 s_0"""



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
print(print_equation(eq_list_1))
eq_list_1 = """f_word
 f_proper
  f_past
   v_0
  s_0
 s_1"""

eq_list_1= """f_join
 f_word
  f_proper
   f_past
    v_0
   s_0
  s_1
 f_proper
  v_1
  s_2"""

eq_list_1= """f_word
 f_cont
  v_0
 s_0"""


print(eq_list_1)
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

sc = input("input your sentence: ").lower()
for item in places:
  sc = sc.replace(item.replace("_", " "), item) 
sc = sc.split(" ")
sd = [part_of_speech[item] for item in sc]
s_1 = list(zip(sd, sc))
if ind_pos(s_1, "question") != -1:
  s_1 = first_person(s_1)
  
def apply_fx(pos, sentence, question_index):
  if question_index != 1:
    return pos
  elif not isinstance(pos[0], tuple):
    return [pos[0]]
  if ind_pos(sentence, "verb") != -1:
    word = replace_tense_word([sentence[ind_pos(sentence, "verb")]], 0)[0][1]
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
  return list_basic
def gen():
  list_basic = []
  list_basic.append([("pronoun", "he"), ("aux", "is")])
  list_basic.append([("pronoun", "i"), ("aux", "am")])
  for item in verb:
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
      final.append(sen)
      
  for item in final:
    if item in list_basic:
      list_basic.remove(item)
  return list_basic

def generate_question(s_1):
  global store
  eq_store = []
  question = []
  for eq in eq_list_1:
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
        store = []
        tmp  = semantic(compute(list(item), list(item2), tree_form(eq)), eq, list(item)+list(item2))
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
