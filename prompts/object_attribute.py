import re
from .prompter import Prompter

class ObjectAttributePrompter(Prompter):

    MESSAGE = [
    {
    "role": "system",
    "content": """Given a question, identify the entities mentioned in the question and list their corresponding attributes. Each entity should be on a new line, and each attribute should be separated by a space. Here is an example:

Sure! Here’s a GPT-3.5 prompt that asks for an analysis of entities and their corresponding attributes from a given question:

---

**Prompt:**

Given a question, identify the entities mentioned in the question and list their corresponding attributes. Each entity should be on a new line, and each attribute should be separated by a space. Here is an example:

**Question:** "Which kind of clothing is not black?"

**Answer:**

```
clothing: color
```

Make sure to maintain the format and accuracy in identifying entities and their attributes for each question.
"""
    },
    {
    "role": "user",
    "content": "**Question**: Who is carrying the umbrella?"
    },
    {
    "role": "assistant",
    "content": """**Answer:**

```
people: position
umbrella: position
```
"""
    },
    {
    "role": "user",
    "content": "**Question**: Which place is it?"
    },
    {
    "role": "assistant",
    "content": """**Answer:**

```
object: usage, scene
```
"""
    },
    {
    "role": "user",
    "content": "**Question**: Does the clothing look large?"
    },
    {
    "role": "assistant",
    "content": """**Answer:**

```
clothing: size
```
"""
    },
    {
    "role": "user",
    "content": "**Question**: Is the chair warm?"
    },
    {
    "role": "assistant",
    "content": """**Answer:**

```
chair: feeling(warm or cold)
```
"""
    },
    ]

    def prompt(self, input):

        prompt_examples = self.MESSAGE
        prompt_examples.append({"role":"user", "content":f"**Question**: {input}"})

        # print(prompt_examples[35]["content"])

        return prompt_examples
    
    def parse(self, output):
        matches = re.findall(r'```([^}]*)```', output)

        if len(matches) > 0:
            return matches[0]
        
        return ""

        