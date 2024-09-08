import os
from PIL import Image
import openai
import numpy as np
import copy
import re
from FlagEmbedding import BGEM3FlagModel

from .step_interpreters import register_step_interpreters, parse_step


class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step,inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state


class ProgramGenerator():
    def __init__(self,prompter,temperature=0.7,top_p=0.5,prob_agg='mean'):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # print("hello")
        # print(os.getenv("OPENAI_API_KEY"))
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0].logprobs):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0].logprobs.token_logprobs[:i]))

    def generate(self,inputs):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=self.prompter.prompt(inputs),
            temperature=self.temperature,
            max_tokens=512,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            logprobs=True
        )

        # print(response.choices[0].logprobs)

        # prob = self.compute_prob(response)
        answer = response.choices[0].message.content

        return self.prompter.parse(answer)

class ProgramSynthesis:

    def __init__(self):
        self.word_close_model = BGEM3FlagModel('BAAI/bge-m3', 
                                               use_fp16=True)

    def parse_objectattribute(self, ObjAttri: str):

        result = {}

        pattern = r'(\w+):\s*([^:\n]+)'
        matches = re.findall(pattern, ObjAttri)

        # 解析匹配结果并填充字典
        for match in matches:
            key = match[0]
            values = match[1].split(', ')
            result[key] = values
        
        return result
    
    def parse_relation(self, Relation: str):
        result = {}

        pattern = r'\(([^)]+)\):\s*([^\n]+)'
        matches = re.findall(pattern, Relation)

        for match in matches:
            objs = match[0].split(', ')
            if(len(objs) == 2):
                key = tuple(objs)
                value = match[1].strip()
                result[key] = value
        
        return result
    
    def find_close(self, word: str, obj_dict: dict):
        candidates = list(obj_dict.keys())

        embeddings_1 = self.word_close_model.encode([word], 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
        embeddings_2 = self.word_close_model.encode(candidates)['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        arr = np.array(similarity[0])

        return candidates[np.argmax(arr)]
    
    def synthesis(self, ObjAttri: str, Relation: str):

        ObjAttriDict = self.parse_objectattribute(ObjAttri)
        RelationDict = self.parse_relation(Relation)

        print(ObjAttriDict)
        print(RelationDict)

        obj_var = {}
        code = ""
        index = 0
        obj_idx = 0

        for obj_name, attributes in ObjAttriDict.items():
            code += f'OBJS{obj_idx}=LOC(image=IMAGE, object="{obj_name}")\n'
            code += f'GRAPH{index}=BUILD(objects=OBJS{obj_idx})\n'

            index += 1
            obj_idx += 1
            
            for attri in attributes:
                code += f'GRAPH{index}=ADD(graph=GRAPH{index-1}, attribute={attri})\n'
                index += 1
            
            obj_var[obj_name] = index - 1

        final_merge_start = index
        
        for (obja, objb), relation in RelationDict.items():
            if obja not in obj_var:
                obja = self.find_close(obja, obj_var)
            if objb not in obj_var:
                objb = self.find_close(objb, obj_var)

            a_idx = obj_var[obja]
            b_idx = obj_var[objb]

            code += f'GRAPH{index}=MERGE(graphA=GRAPH{a_idx}, graphB=GRAPH{b_idx}, relation={relation})\n'

            index += 1
        
        result_index = index - 1
        
        if(len(RelationDict) > 1):
            for i in range(1, len(RelationDict)):
                code += f'GRAPH{index}=MERGE(graphA=GRAPH{index-1}, graphB=GRAPH{final_merge_start+i}, relation={None})\n'
                index += 1

            result_index = index - 1

        code += f"FINAL=RESULT(var=GRAPH{result_index})"

        return code
    