PROMPT = """Think step by step to carry out the instruction.

Instruction: Tag the presidents of US
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='presidents of the US',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the wild animals
Program:
OBJ0=LOC(image=IMAGE,object='wild animal')
LIST0=LIST(query='wild animals',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes with their colors
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='colors',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes by their type
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes (4) by their type
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max=4)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag oscar winning hollywood actors
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='oscar winning hollywood actors',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag these dishes with their cuisines
Program:
OBJ0=LOC(image=IMAGE,object='dish')
LIST0=LIST(query='cuisines',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the utensils used for drinking
Program:
OBJ0=LOC(image=IMAGE,object='utensil')
LIST0=LIST(query='utensils used for drinking',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the logos that have a shade of blue
Program:
OBJ0=LOC(image=IMAGE,object='logo')
LIST0=LIST(query='logos that have a shade of blue',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the logos (10) that have a shade of blue
Program:
OBJ0=LOC(image=IMAGE,object='logo')
LIST0=LIST(query='logos that have a shade of blue',max=10)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag these leaders with the countries they represent
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='countries',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the actor who played Harry Potter
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='actor who played Harry Potter',max=1)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the 7 dwarfs in Snow White
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='dwarfs in snow white',max=7)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {instruction}
Program:
"""