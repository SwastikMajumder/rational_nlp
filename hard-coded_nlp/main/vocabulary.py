def flat(lst):
  output = []
  for item in lst:
    output += item
  return list(set(output))

vocab = {
"wposs": {"he": "his", "you": "your"},
"wobj": {"i": "me", "he": "him"},
"wthird": {"die": "dies", "love": "loves", "kill": "kills", "run": "runs"},
"wpast": {"die": "died", "love": "loved", "kill": "killed", "run": "ran"},
"warticle": {"car-accident": "a-car-accident"}
}

verb_fx = ["wthird", "wpast"]
verb_list = flat([vocab[key] for key in vocab.keys() if key in verb_fx])

verb_fx = ["wthird", "wpast"]
verb_list = flat([vocab[key] for key in vocab.keys() if key in verb_fx])

prep_list = ["at", "to", "on", "in"]
adj_list = ["happy"]

noun_fx = ["warticle"]
noun_list = flat([vocab[key] for key in vocab.keys() if key in noun_fx])+\
            ["wife", "hi", "girl-friend", "homework", "attention", "rock", "shit", "real-name", "name", "age"]

actor_fx = ["wobj", "wposs"]
female_name = ["mary"]
male_name = ["john"]
name = female_name + male_name
actor_list = flat([vocab[key] for key in vocab.keys() if key in actor_fx]) + name + ["everyone"]

key_list = []
for key in vocab.keys():
  key_list += vocab[key].keys()
key_list = list(set(key_list))

def key_ans(vocab, given):
  output = []
  for key in vocab.keys():
    for key2 in vocab[key].keys():
      if key2 == given:
        output.append(vocab[key][key2])
  return output

all_list = {}
for item in key_list:
  all_list[item] = key_ans(vocab, item)

correct = ["what", "is", "are", "a", "by", "am"]
hyphen_word = ["real name"]+[x.replace("-", " ") for x in key_list if "-" in x]
