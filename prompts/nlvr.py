import random

NLVR_CURATED_EXAMPLES=[
"""Statement: An image shows one bare hand with the thumb on the right holding up a belly-first, head-up crab, with water in the background.
Program:
ANSWER0=VQA(image=LEFT,question='Does the image shows one bare hand with the thumb on the right holding a crab?')
ANSWER1=VQA(image=RIGHT,question='Does the image shows one bare hand with the thumb on the right holding a crab?')
ANSWER2=VQA(image=LEFT,question='Is the crab belly-first and head-ups?')
ANSWER3=VQA(image=RIGHT,question='Is the crab belly-first and head-ups?')
ANSWER4=VQA(image=LEFT,question='Is there water in the background?')
ANSWER5=VQA(image=RIGHT,question='Is there water in the background?')
ANSWER6=EVAL(expr='{ANSWER0} and {ANSWER2} and {ANSWER4}')
ANSWER7=EVAL(expr='{ANSWER1} and {ANSWER3} and {ANSWER5}')
ANSWER8=EVAL(expr='{ANSWER6} xor {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)
""",
"""Statement: There is a red convertible in one image.
Program:
ANSWER0=VQA(image=LEFT,question='Is there a red convertible in the image?')
ANSWER1=VQA(image=RIGHT,question='Is there a red convertible in the image?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: One dog is laying down.
Program:
ANSWER0=VQA(image=LEFT,question='How many dogs are laying down?')
ANSWER1=VQA(image=RIGHT,question='How many dogs are laying down?')
ANSWER2=EVAL(expr='{ANSWER0} + {ANSWER1} == 1')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: There are two blue and yellow birds
Program:
ANSWER0=VQA(image=LEFT,question='How many blue and yellow birds are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many blue and yellow birds are in the image?')
ANSWER2=EVAL(expr='{ANSWER0} + {ANSWER1} == 2')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: A single wolf is howling and silhouetted by the moon in one of the images.
Program:
ANSWER0=VQA(image=LEFT,question='How many wolves are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many wolves are in the image?')
ANSWER2=VQA(image=LEFT,question='Is the wolf howling and silhouetted by the moon?')
ANSWER3=VQA(image=RIGHT,question='Is the wolf howling and silhouetted by the moon?')
ANSWER4=EVAL(expr='{ANSWER0} == 1 and {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} == 1 and {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)
""",
"""Statement: One of the two images has a bag with the characters from Disney's Frozen on it.
Program:
ANSWER0=VQA(image=LEFT,question='Does the image have a bag with the characters from Disney's Frozen on it?')
ANSWER1=VQA(Image=RIGHT,question='Does the image have a bag with the characters from Disney's Frozen on it?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: there are at least seven wine bottles in the image on the left
Program:
ANSWER0=VQA(image=LEFT,question='How many wine bottles are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} >= 7')
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: An image shows broccoli growing in soil, with leaves surrounding the florets.
Program:
ANSWER0=VQA(image=LEFT,question='Does the image show broccoli growing in soil?')
ANSWER1=VQA(image=RIGHT,question='Does the image show broccoli growing in soil?')
ANSWER2=VQA(image=LEFT,question='Are leaves surrounding the floret?')
ANSWER3=VQA(image=RIGHT,question='Are leaves surrounding the floret?')
ANSWER4=EVAL(expr='{ANSWER0} and {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} and {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)
""",
"""Statement: An image shows exactly two seals in direct contact, posed face to face.
Program:
ANSWER0=VQA(image=LEFT,question='How many seals are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many seals are in the image?')
ANSWER2=VQA(image=LEFT,question='Are the seals in direct contact?')
ANSWER3=VQA(image=RIGHT,question='Are the seals in direct contact?')
ANSWER4=VQA(image=LEFT,question='Are the seals posed face to face?')
ANSWER5=VQA(image=RIGHT,question='Are the seals posed face to face?')
ANSWER6=EVAL(expr='{ANSWER0} == 2 and {ANSWER2} and {ANSWER4}')
ANSWER7=EVAL(expr='{ANSWER1} == 2 and {ANSWER3} and {ANSWER5}')
ANSWER8=EVAL(expr='{ANSWER6} xor {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)
""",
"""Statement: There is at least two parrots in the right image.
Program:
ANSWER0=VQA(image=RIGHT,question='How many parrots are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} >= 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: In the image on the right, four people are riding in one canoe.
Program:
ANSWER0=VQA(image=RIGHT,question='Are there four people riding in one canoe?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: There are two wolves in each image.
Program:
ANSWER0=VQA(image=LEFT,question='How many wolves are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many wolves are in the image?')
ANSWER2=EVAL(expr='{ANSWER0} == 2 and {ANSWER1} == 2')
FINAL_ANSWER=RESULT(var=ANSWER2)
"""
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='random':
        random.seed(seed)
        prompt_examples = random.sample(NLVR_CURATED_EXAMPLES,num_prompts)
    elif method=='all':
        prompt_examples = NLVR_CURATED_EXAMPLES
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step if the statement is True or False.\n\n{prompt_examples}'
    return prompt_examples + "\nStatement: {statement}\nProgram:".format(**inputs)