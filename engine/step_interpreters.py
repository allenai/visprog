import cv2
import os
import torch
import sys
import timeit
from typing import Union
from openai import OpenAI
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    AutoModelForZeroShotObjectDetection, DetrImageProcessor, DetrForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
from .object_graph import ObjectGraph, ObjectGraphGroup
import re
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
            create_positive_map_label_to_token_from_positive_map

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks


def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'


def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'

    
class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def html(self,eval_expression,step_input,step_output,output_var):
        eval_expression = eval_expression.replace('{','').replace('}','')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def execute(self,prog_step,inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value
        
        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output


class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,question,output_var

    def predict(self,img,question):
        encoding = self.processor(img,question,return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def html(self,img,question,answer,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        answer = html_output(answer)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        question_arg = html_arg_name('question')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;{question_arg}='{question}')={answer}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,question,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        answer = self.predict(img,question)
        prog_step.state[output_var] = answer
        if inspect:
            html_str = self.html(img, question, answer, output_var)
            return answer, html_str

        return answer

class BuildInterpreter():
    step_name = "BUILD"
    
    def __init__(self):
        self.vqa = VQAInterpreter()
        
        self.crop = CropInterpreter()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        objs_var = parse_result['args']['objects']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return objs_var,output_var
    
    def execute(self, prog_step,inspect=False):
        objs_var, output_var = self.parse(prog_step)

        objs = prog_step.state[objs_var]
        img = objs["img"]
        category = objs["category"]

        graphs = ObjectGraphGroup(img)

        for obj in objs["box"]:
            croped_img = self.crop.cropbox(obj, img)
            
            answer = self.vqa.predict(croped_img, f"What's this {category}?")
            
            mid_w, mid_h = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
            w, h = img.size
            location = (mid_w / w, mid_h / h)
            size = (obj[2] - obj[0]), obj[3] - obj[1]

            graphs.add_graph(ObjectGraph(answer, obj, category, location, size))

        print(graphs.Graphs)
        prog_step.state[output_var] = graphs

class ADDInterpreter():
    step_name = 'ADD'

    def __init__(self):
        self.vqa = VQAInterpreter()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        graph_var = parse_result['args']['graph']
        attribute_str = parse_result['args']['attribute']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return graph_var,attribute_str,output_var
    
    def execute(self, prog_step,inspect=False):
        graph_var, attribute_str, output_var = self.parse(prog_step)

        graph = prog_step.state[graph_var]
        img = graph.Img

        res_graph = ObjectGraphGroup(img=img)

        for obj in graph.Graphs:
            if attribute_str in obj.Attribute:
                pass
            else:
                cropped_img = img.crop(obj.Attribute["Box"])
                obj_name = obj.Attribute["Name"]
                answer = self.vqa.predict(cropped_img, f"What's the {attribute_str} of this {obj_name}?")

                obj.add(attribute_str, answer)
            
            res_graph.add_graph(obj)

        prog_step.state[output_var] = res_graph
        
        return res_graph

class MERGEInterpreter():
    step_name = 'MERGE'

    RELATION_MESSAGE = [
    {
    "role": "system",
    "content": """Given the relationship (subject, object): relationship, generate a question that asks about this relationship.

For example:
**Input**: (bottles, wine): right_of 
**Output**: "Is the bottles to the right of the wine?"


3.  should be converted to "Is the book under the table?"

Please follow this format to create the questions.
"""
    },
    {
    "role": "user",
    "content": """**Input:** (people, umbrella): carry"""
    },
    {
    "role": "assistant",
    "content": """**Output:** "Is the people carrying the umbrella?" """
    },
    {
    "role": "user",
    "content": """**Input:** (book, table): under"""
    },
    {
    "role": "assistant",
    "content": """**Output:** "Is the book under the table?" """
    }
    ]

    def __init__(self):
        self.vqa = VQAInterpreter()
        self.client = OpenAI()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        graph1_var = parse_result['args']['graphA']
        graph2_var = parse_result['args']['graphB']
        relation_str = parse_result['args']['relation']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return graph1_var,graph2_var,relation_str,output_var
    
    def llm_ask(self, relation):
        message = self.RELATION_MESSAGE
        message.append({"role": "user", "content": f"**Input:** {relation}"})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.8,
            messages=message
        )

        answer = response.choices[0].message.content

        matches = re.findall(r'"([^}]*)"', answer)

        if len(matches) > 0:
            return matches[0]

        return ""
    
    def get_categories(self, group):
        categories = []

        for graph in group.Graphs:
            if graph.Attribute["Category"] not in categories:
                categories.append(graph.Attribute["Category"])

        return categories

    def focus_image(self, img, box1, box2):
        black_image = Image.new('RGB', img.size, (0, 0, 0))

        region1 = img.crop(box1)
        black_image.paste(region1, (box1[0], box1[1]))

        region2 = img.crop(box2)
        black_image.paste(region2, (box2[0], box2[1]))

        return black_image

    def execute(self, prog_step,inspect=False):
        graph1_var, graph2_var, relation_str, output_var = self.parse(prog_step)

        graph1 = prog_step.state[graph1_var]
        graph2 = prog_step.state[graph2_var]

        if(relation_str == "None"):

            merged_graph = ObjectGraphGroup(groupA=graph1, groupB=graph2)
        
        else:
            merged_graph = ObjectGraphGroup(groupA=graph1, groupB=graph2)
            img = merged_graph.Img

            for objA in graph1.Graphs:
                for objB in graph2.Graphs:
                    new_img = self.focus_image(img, objA.Attribute["Box"], objB.Attribute["Box"])
                    new_img.save("test.jpg")

                    Aname = objA.Attribute["Name"]
                    Bname = objB.Attribute["Name"]

                    question = self.llm_ask(f"({Aname}, {Bname}): {relation_str}")
                    print(question)

                    answer = self.vqa.predict(new_img, question)

                    if(answer == "yes"):
                        merged_graph.add_relation(objA, objB, relation_str)
            
        prog_step.state[output_var] = merged_graph

        return merged_graph
    
class HiddenPrints:
    hide_prints = False

    def __init__(self, model_name=None, console=None, use_newline=True):
        self.model_name = model_name
        self.console = console
        self.use_newline = use_newline
        self.tqdm_aux = None

    def __enter__(self):
        if self.hide_prints:
            import tqdm  # We need to do an extra step to hide tqdm outputs. Does not work in Jupyter Notebooks.

            def nop(it, *a, **k):
                return it

            self.tqdm_aux = tqdm.tqdm
            tqdm.tqdm = nop

            if self.model_name is not None:
                self.console.print(f'Loading {self.model_name}...')
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            # May not be what we always want, but some annoying warnings end up to stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            sys.stdout = self._original_stderr
            if self.model_name is not None:
                self.console.print(f'{self.model_name} loaded ')
            import tqdm
            tqdm.tqdm = self.tqdm_aux

class GLIPLocInterpreter(GLIPDemo):
    step_name = 'LOC'

    def __init__(self, *args_demo):

        working_dir = f'./pretrained_models/GLIP/'
        
        config_file = working_dir + "configs/glip_Swin_L.yaml"
        weight_file = working_dir + "checkpoints/glip_large_model.pth"

        kwargs = {
            'min_image_size': 800,
            'confidence_threshold': 0.5,
            'show_mask_heatmaps': False
        }

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        from maskrcnn_benchmark.config import cfg

        # manual override some options
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", self.dev])

        with HiddenPrints("GLIP"), torch.cuda.device(self.dev):
            from transformers.utils import logging
            logging.set_verbosity_error()
            GLIPDemo.__init__(self, cfg, *args_demo, **kwargs)
        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0
        self.plus = plus
        self.color = 255

    @torch.no_grad()
    def compute_prediction(self, original_image, original_caption, custom_entity=None):
        image = self.transforms(original_image)
        # image = [image, image.permute(0, 2, 1)]
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.dev)
        # caption
        if isinstance(original_caption, list):

            if len(original_caption) > 40:
                all_predictions = None
                for loop_num, i in enumerate(range(0, len(original_caption), 40)):
                    list_step = original_caption[i:i + 40]
                    prediction_step = self.compute_prediction(original_image, list_step, custom_entity=None)
                    if all_predictions is None:
                        all_predictions = prediction_step
                    else:
                        # Aggregate predictions
                        all_predictions.bbox = torch.cat((all_predictions.bbox, prediction_step.bbox), dim=0)
                        for k in all_predictions.extra_fields:
                            all_predictions.extra_fields[k] = \
                                torch.cat((all_predictions.extra_fields[k],
                                            prediction_step.extra_fields[k] + loop_num), dim=0)
                return all_predictions

            # we directly provided a list of category names
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                caption_string += word
                caption_string += seperation_tokens

            tokenized = self.tokenizer([caption_string], return_tensors="pt")
            # tokens_positive = [tokens_positive]  # This was wrong
            tokens_positive = [[v] for v in tokens_positive]

            original_caption = caption_string
            # print(tokens_positive)
        else:
            tokenized = self.tokenizer([original_caption], return_tensors="pt")
            if custom_entity is None:
                tokens_positive = self.run_ner(original_caption)
            # print(tokens_positive)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map,
                                                                                            plus=self.plus)
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with HiddenPrints():  # Hide some deprecated notices
            predictions = self.model(image_list, captions=[original_caption],
                                        positive_map=positive_map_label_to_token)
        predictions = [o.to(self.cpu_device) for o in predictions]
        # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[-2:]
        # if self.tensor_inputs:
        # else:
        #     height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction

    @staticmethod
    def to_left_right_upper_lower(bboxes):
        return [(bbox[1], bbox[3], bbox[0], bbox[2]) for bbox in bboxes]

    @staticmethod
    def to_xmin_ymin_xmax_ymax(bboxes):
        # invert the previous method
        return [(bbox[2], bbox[0], bbox[3], bbox[1]) for bbox in bboxes]

    @staticmethod
    def prepare_image(image):
        image = image[[2, 1, 0]]  # convert to bgr for opencv-format for glip
        return image

    @torch.no_grad()
    def forward(self, image: torch.Tensor, obj: Union[str, list], return_labels: bool = False,
                confidence_threshold=None):

        if confidence_threshold is not None:
            original_confidence_threshold = self.confidence_threshold
            self.confidence_threshold = confidence_threshold

        # if isinstance(object, list):
        #     object = ' . '.join(object) + ' .' # add separation tokens
        image = self.prepare_image(image)

        # Avoid the resizing creating a huge image in a pathological case
        ratio = image.shape[1] / image.shape[2]
        ratio = max(ratio, 1 / ratio)
        original_min_image_size = self.min_image_size
        if ratio > 10:
            self.min_image_size = int(original_min_image_size * 10 / ratio)
            self.transforms = self.build_transform()

        with torch.cuda.device(self.dev):
            inference_output = self.inference(image, obj)

        bboxes = inference_output.bbox.cpu().numpy().astype(int)
        # bboxes = self.to_left_right_upper_lower(bboxes)

        if ratio > 10:
            self.min_image_size = original_min_image_size
            self.transforms = self.build_transform()

        bboxes = torch.tensor(bboxes)

        # Convert to [left, lower, right, upper] instead of [left, upper, right, lower]
        height = image.shape[-2]
        bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)

        if confidence_threshold is not None:
            self.confidence_threshold = original_confidence_threshold
        if return_labels:
            # subtract 1 because it's 1-indexed for some reason
            return bboxes, inference_output.get_field("labels").cpu().numpy() - 1
        return bboxes
    
    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var
    
    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        
        bboxes = self.forward(img,obj_name)

        box_img = self.box_image(img, bboxes)

        objs = dict(
            box = bboxes,
            category = obj_name,
            img = img
        )

        prog_step.state[output_var] = objs
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return objs, html_str

        return objs

class LocInterpreter():
    step_name = 'LOC'

    def __init__(self, thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')

        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device)
        
        special_model_id = "facebook/detr-resnet-50"
        self.special_processor = DetrImageProcessor.from_pretrained(
            special_model_id, revision="no_timm")
        self.special_model = DetrForObjectDetection.from_pretrained(
            special_model_id, revision="no_timm")

        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        prompt = f"a {obj_name}."

        inputs = self.processor(
            images=img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]]
        )
        
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return []

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        return selected_boxes
    
    def special_predict(self, img):
        inputs = self.special_processor(
            images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.special_model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])
        results = self.special_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        boxes, scores = results["boxes"], results["scores"]
        boxes = boxes.tolist()
        scores = scores.tolist()
        if len(boxes)==0:
            return []

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        return selected_boxes

    def top_box(self,img):
        w,h = img.size        
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1

    def html(self,img,box_img,output_var,obj_name):
        step_name=html_step_name(self.step_name)
        obj_arg=html_arg_name('object')
        img_arg=html_arg_name('image')
        output_var=html_var_name(output_var)
        img=html_embed_image(img)
        box_img=html_embed_image(box_img,300)
        return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"


    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        elif obj_name=='object':
            bboxes = self.special_predict(img)
        else:
            bboxes = self.predict(img,obj_name)

        box_img = self.box_image(img, bboxes)

        objs = dict(
            box = bboxes,
            category = obj_name,
            img = img
        )

        prog_step.state[output_var] = objs
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return objs, html_str

        return objs


class Loc2Interpreter(LocInterpreter):

    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        bboxes = self.predict(img,obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))
        prog_step.state[output_var] = objs

        if inspect:
            box_img = self.box_image(img, bboxes, highlight_best=False)
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return objs


class CountInterpreter():
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return box_var,output_var

    def html(self,box_img,output_var,count):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_img = html_embed_image(box_img)
        output = html_output(count)
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        box_var,output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count = len(boxes)
        prog_step.state[output_var] = count
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(box_img, output_var, count)
            return count, html_str

        return count


class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,box_var,output_var

    def html(self,img,out_img,output_var,box_img):
        img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_img}</div>"""
    
    def cropbox(self, box, img):
        box = self.expand_box(box, img.size)
        out_img = img.crop(box)

        return out_img

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            box = []
            out_img = img

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [cx,0,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            right_box = self.right_of(box, img.size)
        else:
            w,h = img.size
            box = []
            right_box = [int(w/2),0,w-1,h-1]
        
        out_img = img.crop(right_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [0,0,cx,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            left_box = self.left_of(box, img.size)
        else:
            w,h = img.size
            box = []
            left_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(left_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,0,w-1,cy]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            above_box = self.above(box, img.size)
        else:
            w,h = img.size
            box = []
            above_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(above_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,cy,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            below_box = self.below(box, img.size)
        else:
            w,h = img.size
            box = []
            below_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(below_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'


class SegmentInterpreter():
    step_name = 'SEG'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            "facebook/maskformer-swin-base-coco")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def pred_seg(self,img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
        instance_map = outputs['segmentation'].cpu().numpy()
        objs = []
        print(outputs.keys())
        for seg in outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            mask = (instance_map==inst_id).astype(float)
            resized_mask = np.array(
                Image.fromarray(mask).resize(
                    img.size,resample=Image.BILINEAR))
            Y,X = np.where(resized_mask>0.5)
            x1,x2 = np.min(X), np.max(X)
            y1,y2 = np.min(Y), np.max(Y)
            num_pixels = np.sum(mask)
            objs.append(dict(
                mask=resized_mask,
                category=category,
                box=[x1,y1,x2,y2],
                inst_id=inst_id
            ))

        return objs

    def html(self,img_var,output_var,output):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        img_arg = html_arg_name('image')
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.pred_seg(img)
        prog_step.state[output_var] = objs
        if inspect:
            labels = [str(obj['inst_id'])+':'+obj['category'] for obj in objs]
            obj_img = vis_masks(img, objs, labels)
            html_str = self.html(img_var, output_var, obj_img)
            return objs, html_str

        return objs


class SelectInterpreter():
    step_name = 'SELECT'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = eval(parse_result['args']['query']).split(',')
        category = eval(parse_result['args']['category'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query,category,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu().numpy()
            
        obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids]

    def html(self,img_var,obj_var,query,category,output_var,output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        category_arg = html_arg_name('category')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{query_arg}={query},{category_arg}={category})={output}</div>"""

    def query_string_match(self,objs,q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q,f'{q}-merged',f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category']==cat]
        
        return None

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,query,category,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        select_objs = []

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs


        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue
                
                select_objs += matches

        if query is not None and len(select_objs)==0:
            select_objs = self.query_obj(query, objs, img)

        prog_step.state[output_var] = select_objs
        if inspect:
            select_obj_img = vis_masks(img, select_objs)
            html_str = self.html(img_var, obj_var, query, category, output_var, select_obj_img)
            return select_objs, html_str

        return select_objs


class ColorpopInterpreter():
    step_name = 'COLORPOP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        gimg = img.copy()
        gimg = gimg.convert('L').convert('RGB')
        gimg = np.array(gimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            gimg = mask*img + (1-mask)*gimg

        gimg = np.array(gimg).astype(np.uint8)
        gimg = Image.fromarray(gimg)
        prog_step.state[output_var] = gimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, gimg)
            return gimg, html_str

        return gimg


class BgBlurInterpreter():
    step_name = 'BGBLUR'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def smoothen_mask(self,mask):
        mask = Image.fromarray(255*mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius = 5))
        return np.array(mask).astype(float)/255

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        bgimg = img.copy()
        bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius = 2))
        bgimg = np.array(bgimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            mask = self.smoothen_mask(mask)
            bgimg = mask*img + (1-mask)*bgimg

        bgimg = np.array(bgimg).astype(np.uint8)
        bgimg = Image.fromarray(bgimg)
        prog_step.state[output_var] = bgimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, bgimg)
            return bgimg, html_str

        return bgimg


class FaceDetInterpreter():
    step_name = 'FACEDET'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.model = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def box_image(self,img,boxes):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            draw.rectangle(box,outline='blue',width=5)

        return img1

    def enlarge_face(self,box,W,H,f=1.5):
        x1,y1,x2,y2 = box
        w = int((f-1)*(x2-x1)/2)
        h = int((f-1)*(y2-y1)/2)
        x1 = max(0,x1-w)
        y1 = max(0,y1-h)
        x2 = min(W,x2+w)
        y2 = min(H,y2+h)
        return [x1,y1,x2,y2]

    def det_face(self,img):
        with torch.no_grad():
            faces = self.model.detect(np.array(img))
        
        W,H = img.size
        objs = []
        for i,box in enumerate(faces):
            x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
            x1,y1,x2,y2 = self.enlarge_face([x1,y1,x2,y2],W,H)
            mask = np.zeros([H,W]).astype(float)
            mask[y1:y2,x1:x2] = 1.0
            objs.append(dict(
                box=[x1,y1,x2,y2],
                category='face',
                inst_id=i,
                mask = mask
            ))
        return objs

    def html(self,img,output_var,objs):
        step_name = html_step_name(self.step_name)
        box_img = self.box_image(img, [obj['box'] for obj in objs])
        img = html_embed_image(img)
        box_img = html_embed_image(box_img,300)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({img_arg}={img})={box_img}</div>"""


    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.det_face(img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(img, output_var, objs)
            return objs, html_str

        return objs


class EmojiInterpreter():
    step_name = 'EMOJI'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        emoji_name = eval(parse_result['args']['emoji'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,emoji_name,output_var

    def add_emoji(self,objs,emoji_name,img):
        W,H = img.size
        emojipth = os.path.join(EMOJI_DIR,f'smileys/{emoji_name}.png')
        for obj in objs:
            x1,y1,x2,y2 = obj['box']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            s = (y2-y1)/1.5
            x_pos = (cx-0.5*s)/W
            y_pos = (cy-0.5*s)/H
            emoji_size = s/H
            emoji_aug = imaugs.OverlayEmoji(
                emoji_path=emojipth,
                emoji_size=emoji_size,
                x_pos=x_pos,
                y_pos=y_pos)
            img = emoji_aug(img)

        return img

    def html(self,img_var,obj_var,emoji_name,output_var,img):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        emoji_arg = html_arg_name('emoji')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img = html_embed_image(img,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{emoji_arg}='{emoji_name}')={img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,emoji_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.add_emoji(objs, emoji_name, img)
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, obj_var, emoji_name, output_var, img)
            return img, html_str

        return img


class ListInterpreter():
    step_name = 'LIST'

    prompt_template = """
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:"""

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.client = OpenAI()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        text = eval(parse_result['args']['query'])
        list_max = eval(parse_result['args']['max'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return text,list_max,output_var

    def get_list(self,text,list_max):
        response = self.client.Completion.create(
            model="text-davinci-002",
            prompt=self.prompt_template.format(list_max=list_max,text=text),
            temperature=0.7,
            max_tokens=256,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )

        item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
        return item_list

    def html(self,text,list_max,item_list,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        max_arg = html_arg_name('max')
        output = html_output(item_list)
        return f"""<div>{output_var}={step_name}({query_arg}='{text}', {max_arg}={list_max})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        text,list_max,output_var = self.parse(prog_step)
        item_list = self.get_list(text,list_max)
        prog_step.state[output_var] = item_list
        if inspect:
            html_str = self.html(text, list_max, item_list, output_var)
            return item_list, html_str

        return item_list


class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        image_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        category_var = parse_result['args']['categories']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return image_var,obj_var,category_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        if len(objs)==0:
            images = [img]
            return []
        else:
            images = [img.crop(obj['box']) for obj in objs]

        if len(query)==1:
            query = query + ['other']

        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            sim = self.calculate_sim(inputs)
            

        # if only one query then select the object with the highest score
        if len(query)==1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class']=query[0]
            obj['class_score'] = 100.0*scores[obj_ids[0],0]
            return [obj]

        # assign the highest scoring class to each object but this may assign same class to multiple objects
        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)
        for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i,cat_id]
            obj['class'] = class_name #+ f'({score_str})'
            obj['class_score'] = round(class_score*100,1)

        # sort by class scores and then for each class take the highest scoring object
        objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
        objs = [obj for obj in objs if 'class' in obj]
        classes = set([obj['class'] for obj in objs])
        new_objs = []
        for class_name in classes:
            cls_objs = [obj for obj in objs if obj['class']==class_name]

            max_score = 0
            max_obj = None
            for obj in cls_objs:
                if obj['class_score'] > max_score:
                    max_obj = obj
                    max_score = obj['class_score']

            new_objs.append(max_obj)

        return new_objs

    def html(self,img_var,obj_var,objs,cat_var,output_var):
        step_name = html_step_name(self.step_name)
        output = []
        for obj in objs:
            output.append(dict(
                box=obj['box'],
                tag=obj['class'],
                score=obj['class_score']
            ))
        output = html_output(output)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        cat_var = html_var_name(cat_var)
        obj_var = html_var_name(obj_var)
        img_arg = html_arg_name('image')
        cat_arg = html_arg_name('categories')
        return f"""<div>{output_var}={step_name}({img_arg}={img_var},{cat_arg}={cat_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        image_var,obj_var,category_var,output_var = self.parse(prog_step)
        img = prog_step.state[image_var]
        objs = prog_step.state[obj_var]
        cats = prog_step.state[category_var]
        objs = self.query_obj(cats, objs, img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(image_var,obj_var,objs,category_var,output_var)
            return objs, html_str

        return objs


class TagInterpreter():
    step_name = 'TAG'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def tag_image(self,img,objs):
        W,H = img.size
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
        for i,obj in enumerate(objs):
            box = obj['box']
            draw.rectangle(box,outline='green',width=4)
            x1,y1,x2,y2 = box
            label = obj['class'] + '({})'.format(obj['class_score'])
            if 'class' in obj:
                w,h = font.getsize(label)
                if x1+w > W or y2+h > H:
                    draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
                    draw.text((x1,y2-h),label,fill='white',font=font)
                else:
                    draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
                    draw.text((x1,y2),label,fill='white',font=font)
        return img1

    def html(self,img_var,tagged_img,obj_var,output_var):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        tagged_img = html_embed_image(tagged_img,300)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('objects')
        output_var = html_var_name(output_var)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var}, {obj_arg}={obj_var})={tagged_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        original_img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.tag_image(original_img, objs)
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, img, obj_var, output_var)
            return img, html_str

        return img


def dummy(images, **kwargs):
    return images, False

class ReplaceInterpreter():
    step_name = 'REPLACE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        device = "cuda"
        model_name = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = dummy

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        prompt = eval(parse_result['args']['prompt'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,prompt,output_var

    def create_mask_img(self,objs):
        mask = objs[0]['mask']
        mask[mask>0.5] = 255
        mask[mask<=0.5] = 0
        mask = mask.astype(np.uint8)
        return Image.fromarray(mask)

    def merge_images(self,old_img,new_img,mask):
        print(mask.size,old_img.size,new_img.size)

        mask = np.array(mask).astype(np.float)/255
        mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
        img = mask*np.array(new_img) + (1-mask)*np.array(old_img)
        return Image.fromarray(img.astype(np.uint8))

    def resize_and_pad(self,img,size=(512,512)):
        new_img = Image.new(img.mode,size)
        thumbnail = img.copy()
        thumbnail.thumbnail(size)
        new_img.paste(thumbnail,(0,0))
        W,H = thumbnail.size
        return new_img, W, H

    def predict(self,img,mask,prompt):
        mask,_,_ = self.resize_and_pad(mask)
        init_img,W,H = self.resize_and_pad(img)
        new_img = self.pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            # strength=0.98,
            guidance_scale=7.5,
            num_inference_steps=50 #200
        ).images[0]
        return new_img.crop((0,0,W-1,H-1)).resize(img.size)

    def html(self,img_var,obj_var,prompt,output_var,output):
        step_name = html_step_name(img_var)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        prompt_arg = html_arg_name('prompt')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var},{prompt_arg}='{prompt}')={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,prompt,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        mask = self.create_mask_img(objs)
        new_img = self.predict(img, mask, prompt)
        prog_step.state[output_var] = new_img
        if inspect:
            html_str = self.html(img_var, obj_var, prompt, output_var, new_img)
            return new_img, html_str
        return new_img


def register_step_interpreters(dataset='nlvr'):
    if dataset=='nlvr':
        return dict(
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='gqa':
        return dict(
            LOC=LocInterpreter(),
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='imageEdit':
        return dict(
            FACEDET=FaceDetInterpreter(),
            SEG=SegmentInterpreter(),
            SELECT=SelectInterpreter(),
            COLORPOP=ColorpopInterpreter(),
            BGBLUR=BgBlurInterpreter(),
            REPLACE=ReplaceInterpreter(),
            EMOJI=EmojiInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='okDet':
        return dict(
            FACEDET=FaceDetInterpreter(),
            LIST=ListInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            RESULT=ResultInterpreter(),
            TAG=TagInterpreter(),
            LOC=Loc2Interpreter(thresh=0.05,nms_thresh=0.3)
        )
    elif dataset=='graph':
        return dict(
            LOC=GLIPLocInterpreter(),
            BUILD=BuildInterpreter(),
            ADD=ADDInterpreter(),
            MERGE=MERGEInterpreter(),
            RESULT=ResultInterpreter()
        )