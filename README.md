<br>
<b> the meaning engine giving the following output </b> <br>
```
S(past(verb3(john,kill,mary)))
S(past(verb3(john,escape,the-police)))
S(declare(john,crook))
S(past(verb2(mary,die)))
S(that(verb2(the-police,know),past(verb2(mary,die))))
S(that(verb2(john,know),past(verb3(john,kill,mary))))
S(that(verb2(john,know),declare(john,crook)))
S(that(verb2(john,know),past(verb2(mary,die))))
S(past(verb3(john,kill,mary)))
S(that(verb2(the-police,know),past(verb2(mary,die))))
S(that(verb2(john,know),past(verb3(john,kill,mary))))
S(comma(past(verb3(the-police,ask,john)),who(past(verb2(mary,kill)))))
S(past(verb3(john,kill,mary)))
S(that(verb2(the-police,know),past(verb2(mary,die))))
S(that(verb2(john,know),past(verb3(john,kill,mary))))
S(comma(past(verb3(the-police,ask,john)),who(past(verb2(mary,kill)))))
S(comma(past(verb3(the-police,ask,john)),who(past(verb2(mary,kill)))))
S(that(past(verb3(john,tell,the-police)),past(verb3(john,kill,mary))))
S(that(verb2(the-police,know),past(verb3(john,kill,mary))))
S(that(verb2(the-police,know),declare(john,crook)))
S(verb3(the-police,punish,john))
S(that(verb2(the-police,know),past(verb2(mary,die))))
S(that(past(verb3(john,tell,the-police)),past(verb2(mary,die))))
S(that(past(verb3(john,tell,the-police)),declare(john,crook)))

the police knows that mary died
john knows that john killed mary
john knows that john is a crook
mary died
john told the police that john killed mary
john killed mary
john told the police that john is a crook
the police asked john ," who killed mary ?"
john escaped the police
the police knows that john killed mary
john told the police that mary died
john is a crook
the police knows that john is a crook
the police punishes john
john knows that mary died
```
<b> the grammar engines</b>
run the main.py files to get this output - <br>
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
