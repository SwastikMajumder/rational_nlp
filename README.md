<b> the main.py file in hard-coded's main directory (use this one): </b> <br>

run the main.py files afterwards to get this output - <br>
```
*******
inputted sentence: he died in a car accident
words of processed sentence: he, die, car-accident, a, in

tried equations:

past(verb2(he,die))
ask(past(verb2(he,die)))
obj(past(verb2(he,die)),poss(he,car-accident))
obj(prep(past(verb2(poss(he,car-accident),die)),in),he)
obj(prep(verb2(poss(he,car-accident),die),in),he)
verb2(he,die)
noun(prep(past(verb2(he,die)),in),car-accident)
noun(prep(verb2(he,die),in),car-accident)

he died
did he die
he died his car accident
his car accident died in him
his car accident dies in him
he dies
he died in a car accident
he dies in a car accident

final answer: 
noun(prep(past(verb2(he,die)),in),car-accident)
*******
```
this is converting the sentence "he died in a car accident" into its algebraic form. <br>
<br>
