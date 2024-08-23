eq_list = """f_past
 f_verb2
  actor_0
  verb_0

f_ask
 f_past
  f_verb2
   actor_0
   verb_0

f_obj
 f_past
  f_verb2
   actor_0
   verb_0
 f_poss
  actor_0
  noun_0

f_obj
 f_prep
  f_past
   f_verb2
    f_poss
     actor_0
     noun_0
    verb_0
  prep_0
 actor_0

f_obj
 f_prep
  f_verb2
   f_poss
    actor_0
    noun_0
   verb_0
  prep_0
 actor_0

f_prep
 f_past
  f_verb3
   actor_0
   verb_0
   noun_0
 prep_0
 actor_1

f_by
 f_past
  f_verb3
   actor_0
   verb_0
   actor_1
 f_prep
  f_past
   f_verb3
    actor_0
    verb_1
    noun_0
  prep_0
  actor_1

f_verb2
 actor_0
 verb_0

f_noun
 f_prep
  f_past
   f_verb2
    actor_0
    verb_0
  prep_0
 noun_0

f_noun
 f_prep
  f_verb2
   actor_0
   verb_0
  prep_0
 noun_0

f_prep
 f_past
  f_verb2
   actor_0
   verb_0
 prep_0
 actor_1

f_verb3
 actor_0
 verb_0
 actor_1

f_because
 f_verb3
  actor_0
  verb_0
  actor_1
 f_verb3
  actor_1
  verb_0
  actor_0

f_past
 f_verb3
  actor_0
  verb_0
  actor_1"""
import vocabulary
def return_equation(sentence):
    pos_dic = {"verb": vocabulary.verb_list, "actor": vocabulary.actor_list, "noun": vocabulary.noun_list, "adj": vocabulary.adj_list, "prep": vocabulary.prep_list}
    vocab = vocabulary.all_list
    number = {"0": "verb", "1": "actor", "2": "noun", "3": "adj", "4": "prep"}
    #sentence = "mary threw a rock"
    correct = vocabulary.correct
    hyphen_word = vocabulary.hyphen_word
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
    orig = sentence
    for item in hyphen_word:
        sentence = sentence.replace(item, item.replace(" ", "-"))
    sentence = sentence.split()
    new_sentence = []
    for item in sentence:
      if item in correct:
        new_sentence.append(item)
      elif item in change:
        for key in vocab.keys():
          if item in vocab[key]:
            new_sentence.append(key)
    sentence = orig.split()
    new_sentence = list(set(new_sentence))
    print()
    print("*******")
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
      l_verb = list(itertools.permutations(fix_empty_list(pos_dic["verb"], max_key(equation, "verb")), max_key(equation, "verb")))
      l_actor = list(itertools.permutations(fix_empty_list(pos_dic["actor"], max_key(equation, "actor")), max_key(equation, "actor")))
      l_noun = list(itertools.permutations(fix_empty_list(pos_dic["noun"], max_key(equation, "noun")), max_key(equation, "noun")))
      l_adj = list(itertools.permutations(fix_empty_list(pos_dic["adj"], max_key(equation, "adj")), max_key(equation, "adj")))
      l_prep = list(itertools.permutations(fix_empty_list(pos_dic["prep"], max_key(equation, "prep")), max_key(equation, "prep")))
      for item in itertools.product(l_verb, l_actor, l_noun, l_adj, l_prep):
        #print(item)
        new_eq = equation
        for i in range(len(item)):
          for j in range(len(item[i])):
            new_eq = new_eq.replace(number[str(i)] + "_" + str(j), "d_" + item[i][j])

        if "d_none" not in new_eq:
            input_arr.append(new_eq)
            
    #input_arr = "\n\n".join(input_arr)
    print("\ntried equations:\n")
    import simple_8
    output_arr = simple_8.process(input_arr)

    sentence = " ".join(sentence)
    #input_arr = input_arr.split("\n\n")
    print("\nfinal answer: ")
    for i in range(len(output_arr)):
      if sentence.replace("-"," ").replace(" 's", "'s") == output_arr[i]:
        print(string_equation(input_arr[i]))
        print("*******")
        print()
        return input_arr[i]
return_equation("he died in a car accident")
