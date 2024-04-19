import random

GQA_CURATED_MESSAGES=[
{
"role": "system",
"content": """You are a visual programmer, you need to generate corresponding code according to user's question.
Here are some visual functions:
- LOC(image=, object=): it can locate the specific region of the image and return the region box, which is able to detect object.
- CROP(image=, box=): it can crop the image, fitting into the region box.
- COUNT(box=): it can count the number of region boxes.
- CROP_RIGHTOF(image=,box=): it can crop the image, leaving the right area of the region box.
- CROP_LEFTOF(image=,box=): it can crop the image, leaving the left area of the region box.
- CROP_FRONTOF(image=,box=): it can crop the image, leaving the area in front of the region box.
- CROP_INFRONTOF(image=,box=): it can crop the image, leaving the area in the front of the region box.
- CROP_INFRONT(image=,box=): it can crop the image, leaving the area in front the region box.
- CROP_BEHIND(image=,box=): it can crop the image, leaving the area behind the region box.
- CROP_AHEAD(image=,box=): it can crop the image, leaving the area head of the region box.
- CROP_BELOW(image=,box=): it can crop the image, leaving the area below the region box.
- CROP_ABOVE(image=,box=): it can crop the image, leaving the area above the region box.
- VQA(image=,question=): it can answer the question according to the image.
- RESULT(var=): it can use a variable as the final result.
"""
},
{
"role": "user",
"content": "Is the vehicle in the top of the image?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Are there trains or fences in this scene?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='train')
BOX1=LOC(image=IMAGE,object='fence')
ANSWER0=COUNT(box=BOX0)
ANSWER1=COUNT(box=BOX1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} + {ANSWER1} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER)"""
},
{
"role": "user",
"content": "Who is carrying the umbrella?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='umbrella')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Who is carrying the umbrella?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Which place is it?"
},
{
"role": "assistant",
"content": """ANSWER0=VQA(image=IMAGE,question='Which place is it?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "What color is the curtain that is to the right of the mirror?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='mirror')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the curtain?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Is the pillow in the top part or in the bottom of the picture?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='pillow')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'top' if {ANSWER0} > 0 else 'bottom'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Do you see bottles to the right of the wine on the left of the picture?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='LEFT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='wine')
IMAGE1=CROP_RIGHTOF(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object='bottles')
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Is the street light standing behind a truck?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='truck')
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='street light')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Which side is the food on?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='food')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if {ANSWER0} > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "What do the wetsuit and the sky have in common?"
},
{
"role": "assistant",
"content": """ANSWER0=VQA(image=IMAGE,question='What do the wetsuit and the sky have in common?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Do the post and the sign have a different colors?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='post')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='sign')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='What color is the post?')
ANSWER1=VQA(image=IMAGE1,question='What color is the sign?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} != {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)"""
},
{
"role": "user",
"content": "Does the traffic cone have white color?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='traffic cone')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the traffic cone?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'white' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Are these animals of different species?"
},
{
"role": "assistant",
"content": """ANSWER0=VQA(image=IMAGE,question='Are these animals of different species?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Which side of the image is the chair on?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='chair')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if {ANSWER0} > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Do you see any drawers to the left of the plate?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='plate')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='drawers')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Does the mat have the same color as the sky?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='sky')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='mat')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='What color is the sky?')
ANSWER1=VQA(image=IMAGE1,question='What color is the mat?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)"""
},
{
"role": "user",
"content": "Is a cat above the mat?"
},
{
"role": "assistant",
"content": """BOX0=LOC(image=IMAGE,object='mat')
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='cat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    # if method=='all':
    #     prompt_examples = GQA_CURATED_EXAMPLES
    # elif method=='random':
    #     random.seed(seed)
    #     prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
    # else:
    #     raise NotImplementedError

    

    # prompt_examples = '\n'.join(prompt_examples)
    # prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'

    prompt_examples = GQA_CURATED_MESSAGES
    prompt_examples.append({"role":"user", "content":inputs})

    # print(prompt_examples[35]["content"])

    return prompt_examples