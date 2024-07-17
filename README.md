<b> the hard-coded directory (older and not used anymore): </b> <br>
it was the older versions of the program where machine learning models was not used - but it turns out that that approach is slow to develop because of manually coding all the rules <br>
you can try "i am running fast" with archive_001.py for seeing if it still works <br>
<br>
<b> the ml directory: </b> <br>
the current model which is being used is facebook's bart and not t5 (transformers) <br>
open the bart folder and run the setup_train_model.py, it will create some bulky files which will be needed by the main.py to run <br>
run the main.py file afterwards to get this output - <br>
```
inputted sentence: you gave your attention to me
words of processed sentence: you, give, you, attention, to, i

guessed equations:

past(verb(you,give))
past(verb(i,give))
verb(you,give)
verb(i,give)
prep(past(noun(verb(you,give)),poss(you,attention)),to,you)
prep(past(noun(verb(you,give)),poss(you,attention)),to,i)
prep(past(noun(verb(you,give)),poss(i,attention)),to,you)
prep(past(noun(verb(you,give)),poss(i,attention)),to,i)
prep(past(noun(verb(i,give)),poss(you,attention)),to,you)
prep(past(noun(verb(i,give)),poss(you,attention)),to,i)
prep(past(noun(verb(i,give)),poss(i,attention)),to,you)
prep(past(noun(verb(i,give)),poss(i,attention)),to,i)
prep(noun(past(verb(you,give)),attention),to,you)
prep(noun(past(verb(you,give)),attention),to,i)
prep(noun(past(verb(i,give)),attention),to,you)
prep(noun(past(verb(i,give)),attention),to,i)
ask(poss(you,attention))
ask(poss(i,attention))
past(verb3(you,give,you))
past(verb3(you,give,i))
past(verb3(i,give,you))
past(verb3(i,give,i))

you gave
i gave
you give
i give
you gave your attention to you
you gave your attention to me
you gave my attention to you
you gave my attention to me
i gave your attention to you
i gave your attention to me
i gave my attention to you
i gave my attention to me
you gave attention to you
you gave attention to me
i gave attention to you
i gave attention to me
what is your attention
what is your attention
you gave you
you gave mary
i gave you
i gave i

final answer: 
prep(past(noun(verb(you,give)),poss(you,attention)),to,i)
```
this is converting the sentence "you gave your attention to me" into its algebraic form. <br>
<b> computational intensive to train the models using setup files: </b> <br>
<br>
use google research colab if your computer is slow <br> <br>
<b> language algebra directory </b> <br>
perform algebra with the algebraic form of the sentence once its converted to that form
