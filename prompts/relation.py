import re
from .prompter import Prompter

class RelationPrompter(Prompter):
    MESSAGE = [
    {
    "role": "system",
    "content": """Given a question and a list of objects along with their attributes, identify and describe the relationship between the objects mentioned in the question.

For example, if the question is "Who is carrying the umbrella?" and the objects and attributes are:

```
people: position
umbrella: position
```

The output should be:

```
(people, umbrella): carry
```

This indicates that there is a "carry" relationship between "people" and "umbrella".

**Input:**

1. A question.
2. A list of objects with their associated attributes.

**Output:**

The relationship between the objects mentioned in the question, formatted as (object1, object2): relationship.
    """
    },
    {
    "role": "user",
    "content": """**Question:** "Who is driving the car?"
**Objects and attributes:**
```
person: role
car: object
```"""
    },
    {
    "role": "assistant",
    "content": """**Output:**
```
(person, car): drive
```"""
    },
    {
    "role": "user",
    "content": """**Question:** "What animal is eating the grass?"
**Objects and attributes:**
```
animal: species
grass: object
    ```"""
    },
    {
    "role": "assistant",
    "content": """**Output:**
```
(animal, grass): eat
```"""
    },
    ]

    def prompt(self, input):

        prompt_examples = self.MESSAGE
        Question = input["Question"]
        ObjectAttribute = input["ObjectAttribute"]
        prompt_examples.append({"role":"user", "content":f"**Question**: {Question}\n**Objects and attributes:**\n```\n{ObjectAttribute}\n```"})

        return prompt_examples

    def parse(self, output):
        matches = re.findall(r'```([^}]*)```', output)

        if len(matches) > 0:
            return matches[0]
        
        return ""
    