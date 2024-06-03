import random

GQA_CURATED_MESSAGES=[
{
"role": "system",
"content": """You are a visual programmer, you need to generate corresponding code according to user's question(without picture).
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

REMEMBER: You are not allowed to generate no program. Try to handle with the question step by step.
"""
},
{
"role": "user",
"content": "Question: Is the vehicle in the top of the image?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to locate the top area.
2. Find vehicles.
3. Count the number of vehicles.
4. If the number of vehicles is greater than 0, say "yes".
5. Otherwise, say "no".
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Are there trains or fences in this scene?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the trains.
2. I need to find the fences.
3. Count the number of the trains.
4. Count the number of the fences.
5. Plus the numbers of the trains and the fences.
6. If the sum is greater than 0, say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='train')
BOX1=LOC(image=IMAGE,object='fence')
ANSWER0=COUNT(box=BOX0)
ANSWER1=COUNT(box=BOX1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} + {ANSWER1} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER)"""
},
{
"role": "user",
"content": "Question: Who is carrying the umbrella?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the umbrella.
2. Check out the area of the umbrella.
3. Ask who is carrying the umbrella.
Program:
BOX0=LOC(image=IMAGE,object='umbrella')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Who is carrying the umbrella?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Question: Which place is it?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I don't have any object informations. So I decide to simply ask.
Program:
ANSWER0=VQA(image=IMAGE,question='Which place is it?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Question: What color is the curtain that is to the right of the mirror?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the mirror.
2. Locate the right area of the mirror.
3. Ask the color of the curtain.
4. Mark the color as the answer.
Program:
BOX0=LOC(image=IMAGE,object='mirror')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the curtain?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Question: Is the pillow in the top part or in the bottom of the picture?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I decide to check out the top of the picture.
2. Is there any pillows?
3. If the number of pillows is greater than 0, say "top".
4. Otherwise, say "bottom".
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='pillow')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'top' if {ANSWER0} > 0 else 'bottom'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Do you see bottles to the right of the wine on the left of the picture?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to locate the left of the picture.
2. I need to find the wine on the left of the picture.
3. Check out the right area of the wine on the left of the picture.
4. Is there any bottles?
5. If the number of bottles is greater than 0, say "yes". Otherwise, say "no".
Program:
BOX0=LOC(image=IMAGE,object='LEFT')
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
"content": "Question: Is the street light standing behind a truck?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the truck.
2. Check out the behind of the truck.
3. Count the number of the truck.
4. If the number of the truck is greater than 0, say "yes". Otherwise, say "no".
Program:
BOX0=LOC(image=IMAGE,object='truck')
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='street light')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Which side is the food on?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I decide to check the right side first.
2. Check out is there any food on the right side.
3. If the number of foof on the right side is greater than zero, say "right".
4. Otherwise, "left".
Program:
BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='food')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if {ANSWER0} > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: What do the wetsuit and the sky have in common?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I have no idea of the common attribute, so I decide to simply ask.
Program:
ANSWER0=VQA(image=IMAGE,question='What do the wetsuit and the sky have in common?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Question: Do the post and the sign have a different colors?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the post.
2. I need to find the sign.
3. I need to ask the color of the post.
4. I need to ask the color of the sign.
5. Check the colors between post and sign.
6. If the colors are the same, say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='post')
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
"content": "Question: Does the traffic cone have white color?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the traffic cone.
2. I need to ask the color of the traffic cone.
3. If the color of the traffic cone is white, I say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='traffic cone')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the traffic cone?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'white' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Are these animals of different species?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I can't check every animal, so I decide to simply ask.
Program:
ANSWER0=VQA(image=IMAGE,question='Are these animals of different species?')
FINAL_RESULT=RESULT(var=ANSWER0)"""
},
{
"role": "user",
"content": "Question: Which side of the image is the chair on?"
},
{
"role": "assistant",
"content": """Reasoning:
1. First, I check the right side of the image.
2. Is there any chair?
3. If there is a chair, I say "right". Otherwise, "left".
Program:
BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='chair')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if {ANSWER0} > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Do you see any drawers to the left of the plate?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the plate.
2. I need to look the left of the plate.
3. Is there any drawers?
4. If the number of drawers is greater than 0, I say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='plate')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='drawers')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)"""
},
{
"role": "user",
"content": "Question: Does the mat have the same color as the sky?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the sky.
2. I need to find the mat.
3. I need to know the sky's color.
4. I need to know the mat's color.
5. If the two colors are same, I say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='sky')
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
"content": "Question: Is a cat above the mat?"
},
{
"role": "assistant",
"content": """Reasoning:
1. I need to find the mat.
2. Look above the mat and see if there is a cat.
3. If there is a cat, I say "yes". Otherwise, "no".
Program:
BOX0=LOC(image=IMAGE,object='mat')
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
    prompt_examples.append({"role":"user", "content":f"Question: {inputs}"})

    # print(prompt_examples[35]["content"])

    return prompt_examples