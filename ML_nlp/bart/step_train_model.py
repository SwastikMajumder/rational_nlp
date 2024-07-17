training_data_past = [
    ("john kills mary", "john killed mary"),
    ("who dies", "who died"),
    ("she is running", "she was running"),
    ("he runs", "he ran"),
    ("they are happy", "they were happy"),
    ("john kills", "john killed"),
    ("john kills whom", "john killed whom"),
    ("kill someone", "killed someone"),
    ("why are you crying", "why were you crying"),
    ("what are they doing", "what were they doing"),
    ("does john", "did john"),
    ("does john kill", "did john kill"),
    ("whom does john kill", "whom did john kill"),
    ("who kills", "who killed"),
    ("john cries", "john cried"),
    ("who kills mary", "who killed mary"),
    ("he dies", "he died"),
    ("they are running", "they were running"),
    ("they run", "they ran"),
    ("mary is crying", "mary was crying"),
    ("mary cries", "mary cried"),
    ("she is running", "she was running"),
    ("how are you", "how were you"),
    ("who am i", "who was i"),
    ("who are you", "who were you"),
    ("john runs fast", "john ran fast"),
    ("john is running slowly", "john was running slowly"),
    ("john runs slowly", "john ran slowly"),
    ("mary kills john", "mary killed john"),
    ("what are you doing", "what were you doing"),
    ("what is your name", "what was your name"),
    ("this is my book", "this was my book"),
    ("this one is mine", "this one was mine"),
    ("are you happy", "were you happy"),
    ("she never lies", "she never lied"),
    ("they never understand", "they never understood"),
    ("he understands", "he understood"),
    ("she understands me", "she understood me"),
    ("this is me", "this was me"),
    ("she is killed", "she was killed"),
    ("she never listens to me", "she never listened to me"),
    ("they laugh at me", "they laughed at me"),
    ("john throws a book at mary", "john threw a book at mary"),
    ("john throws a book", "john threw a book"),
    ("is it fine", "was it fine"),
    ("he laughs", "he laughed"),
    ("they laugh", "they laughed"),
    ("what are you here for", "what were you here for"),
    ("do you complete your homework", "did you complete your homework"),
    ("she has done her homework", "she had done her homework"),
    ("they have him quiet", "they had him quiet"),
    ("she is very kind", "she was very kind"),
    ("she does not know me", "she did not know me"),
    ("no one completes their homework", "no one completed their homework"),
    ("he do his homework", "he did his homework"),
    ("he can write his name", "he could write his name"),
    ("they are bothered", "they were bothered"),
    ("what is his name", "what was his name"),
    ("he goes to school", "he went to school"),
    ("she goes to office", "she went to office"),
    ("he is gone", "he was gone"),
    ("he leaves his school", "he left his school"),
    ("they are worried", "they were worried"),
    ("they are kind", "they were kind"),
    ("you are very good", "you were very good"),
    ("they are very nice", "they were very nice"),
    ("he walks very steadily", "he walked very steadily"),
    ("he walks", "he walked"),
    ("she is kind", "she was kind"),
    ("we are nice to him", "we were nice to him"),
    ("john does not know mary", "john did not know mary"),
    ("do you know me", "did you know me"),
    ("i am happy", "i was happy"),
    ("i am sorry", "i was sorry"),
    ("she walks to school", "she walked to school"),
    ("he eats breakfast", "he ate breakfast"),
    ("we play board games on weekends", "we played board games on weekends"),
    ("she drinks coffee every morning", "she drank coffee every morning"),
    ("i am a human", "i was a human"),
    ("i will do it", "i did it"),
    ("she has been doing it", "she had been doing it"),
    ("he drives to work", "he drove to work"),
    ("i write in my journal", "i wrote in my journal"),
    ("she paints beautiful pictures", "she painted beautiful pictures"),
    ("john pushes", "john pushed"),
    ("john throws", "john threw"),
    ("mary throws", "mary threw"),
    ("you give", "you gave")
]

training_data_verb = [
    ("john + run", "john runs"),
    ("i + run", "i run"),
    ("they + go", "they go"),
    ("he + go", "he goes"),
    ("mary + want", "mary wants"),
    ("he + eat", "he eats"),
    ("they + run", "they run"),
    ("we + sing", "we sing"),
    ("john + push", "john pushes"),
    ("john + throw", "john throws")
]

training_data_noun = [
    ("john threw + rock", "john threw a rock"),
    ("john painted + wall", "john painted the wall"),
    ("i + human", "i am a human"),
    ("i am good + boy", "i am a good boy"),
    ("i broke + chair", "i broke the chair"),
    ("john is good + man", "john is a good man"),
    ("you gave + your attention", "you gave your attention")
]

training_data_verb3 = [
    ("john + kill + mary", "john kills mary"),
    ("john + annoy + mary", "john annoys mary"),
    ("john + throw + mary", "john throws mary"),
    ("mary + throw + mary", "mary throws mary"),
    ("i + give + you", "i give you")
]

training_data_by = [
    ("john killed mary + john threw a rock at mary", "john killed mary by throwing a rock at mary"),
    ("john painted the wall + john used a brush on the wall", "john painted the wall by using a brush on the wall"),
    ("john finished the race + john ran across the finishing line", "john finished the race by running across the finishing line")
]

training_data_prep = [
    ("talk + with + i", "talk with me"),
    ("we walk + to + school", "we walk to school"),
    ("birds fly + in + sky", "birds fly in the sky"),
    ("john was going + to + john's home", "john was going to john's home"),
    ("cat + on + mat", "the cat is on the mat"),
    ("she sat + under + tree", "she sat under the tree"),
    ("he + at + store", "he is at the store"),
    ("book + on + table", "the book is on the table"),
    ("we went + through + tunnel", "we went through the tunnel"),
    ("he stood + beside + she", "he stood beside her"),
    ("keys + in + drawer", "the keys are in the drawer"),
    ("john threw a rock + at + mary", "john threw a rock at mary"),
    ("john used a brush + on + wall", "john used a brush on the wall"),
    ("john ran + across + finishing line", "john ran across the finishing line"),
    ("you gave your attention + to + i", "you gave your attention to me")
]

training_data_poss = [
    ("you + name", "your name"),
    ("john + car", "john's car"),
    ("you + friend", "your friend"),
    ("you + real name", "your real name"),
    ("mary + hand", "mary's hand"),
    ("i + book", "my book"),
    ("mary + age", "mary's age"),
    ("you + attention", "your attention")
]

training_data_ask = [
    ("your name", "what is your name"),
    ("your real name", "what is your real name"),
    ("you", "how are you"),
    ("your favourite ice cream", "what is your favourite ice cream"),
    ("john", "how is john"),
    ("you are happy", "are you happy"),
    ("mary's age", "what is mary's age")
]

training_data_adj = [
    ("you + happy", "you are happy"),
    ("i + sorry", "i am sorry"),
    ("he + honest", "he is honest"),
    ("john + good", "john is good"),
    ("they + talented", "they are talented"),
    ("we + heroic", "we are heroic")
]

import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW

names = ["past"]
training_data_set = [training_data_past]
count = 0

for training_data in training_data_set:
    # Initialize the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    # Prepare the input and output sequences
    input_sequences = [input_text for input_text, output_text in training_data]
    output_sequences = [output_text for input_text, output_text in training_data]

    # Tokenize the sequences
    input_tokens = tokenizer(input_sequences, return_tensors="pt", padding=True, truncation=True)
    output_tokens = tokenizer(output_sequences, return_tensors="pt", padding=True, truncation=True)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids=input_tokens.input_ids, attention_mask=input_tokens.attention_mask, labels=output_tokens.input_ids)
        # Compute loss
        loss = outputs.loss
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    count += 1
    torch.save(model.state_dict(), 'bart_model_' + names[count-1] + '.pth')
    print(f'{count}/{len(training_data_set)}')
    print()

print("Training completed.")


"""
# Testing the model
test_input = "john opened the door + john turned the key in the lock"
test_input_tokens = tokenizer(test_input, return_tensors="pt")

# Generate the output sequence
model.eval()
with torch.no_grad():
    generated_ids = model.generate(test_input_tokens.input_ids, max_length=50, num_beams=5, early_stopping=True)
    predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Input Sequence:")
print(test_input)
print("\nPredicted Completion:")
print(predicted_text)
"""
