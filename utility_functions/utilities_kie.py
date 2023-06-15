from Template import Template
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas as Canvas
import cv2
import matplotlib.pyplot as plt 
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from pdf2image import convert_from_path
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
import os
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import numpy as np
from reportlab.lib.pagesizes import A4
import random
import pytesseract
from pytesseract import Output
import cv2
from transformers import AutoTokenizer, LayoutLMv2ForRelationExtraction, AutoModel
import os
import numpy as np
import time
import copy
import string
import editdistance


def unify_keys_vals(dict_):
    unified_dict = {}
    for key_1 in dict_.keys():
        if 'config' not in key_1:
            for key_2 in dict_[key_1].keys():
                element = dict_[key_1][key_2]
                element = str(element)
                #key_1 = str(key_1).lower()
                key_2 = str(key_2).lower()
                unified_dict[key_2] = element.lower()
    return unified_dict
    
def get_ocr_data(conf_val:float=50, image_path:str= 'form.jpg'):
    image = cv2.imread(image_path)
    results = pytesseract.image_to_data(image, output_type=Output.DICT, lang='deu')
    n_boxes = len(results['level'])
    tokens = []
    bboxes = []
        
    for i in range(n_boxes):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
    	# extract the OCR text itself along with the confidence of the
	    # text localization
        text = results["text"][i]
    
        conf = int(results["conf"][i])
        #(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if conf > conf_val:
        # display the confidence and text to our terminal
        #tokens.append((text, [x + w, y + h]))
            tokens.append(text)
            bboxes.append([x,y,x+w, y+h])
            # strip out non-ASCII text so we can draw the text on the image
		    # using OpenCV, then draw a bounding box around the text along
		    # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
    return tokens, bboxes, image
    
def add_tokens_tokenizer( tokens:list=None, bboxes:list=None, tokenizer:AutoTokenizer=None, model:LayoutLMv2ForRelationExtraction=None):
    
    new_tokens = tokens

    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))
        
    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    #tokenizer.train_new_from_iterator(tokenizer.vocab, 1000)

    #model_2 = copy.deepcopy(model)
    #tokenizer_2 = copy.deepcopy(tokenizer)
       
    return tokenizer, model
        

def preprocess_tokens(tokens:list=None, bboxes:list=None):
    speical_characters = set(string.punctuation)
    tokens_preprocessed = []
    bboxes_preprocessed = []
        
    for i, token in enumerate(tokens):
        token = token.lower()
        for char_ in speical_characters:
            token = token.strip(char_)
        if token not in speical_characters and token not in (' ',  '', '\t','\n') and len(token) > 1:
            tokens_preprocessed.append(token)
            bboxes_preprocessed.append(bboxes[i])
        
    return tokens_preprocessed, bboxes_preprocessed

def encode_tokens(tokens:list=None, bboxes:list = None, tokenizer : AutoTokenizer = None) :
    input_ids = []
    bboxes_tokenized = []
    input_id_map ={}
    input_id_index_map = {}
    
    for i in range(len(tokens)):
        tokenized_tokens = tokenizer.tokenize(tokens[i])
        #print(tokenized_token)
        #print(bboxes[i])
        input_ids_local = tokenizer.encode(text=tokenized_tokens, boxes = [bboxes[i] for j in range(len(tokenized_tokens))], is_pretokenized=True, add_special_tokens=False)
        for tokenized_token, input_id in zip(tokenized_tokens, input_ids_local):
            if tokenized_token not in set(str(string.punctuation)) and tokenized_token not in ('', ' ', '\t', '\n'):
                input_ids.append(input_id)
                input_id_map[tokenized_token] = input_id
                bboxes_tokenized.append(bboxes[i]) 
    return input_ids, bboxes_tokenized, input_id_map    
    
    
def check_token_presence_within_list(token:str=None, token_list:str=None):
    flag = False
    for i in token_list:
        if token in i:
            flag = True
            return flag
    return flag

def rearrange_token_order(token_list :list =None, token_key_str:str=None, token_indices:list=None):
    indices = {}
    for token in token_list:
        indices[token_key_str.index(token)]= token
    
    sorted_indices = sorted([key for key in indices.keys()])
    token_indices = {indices[key]:i for i,key in enumerate(sorted_indices)}
        
    for token1 in token_list:
        for token2 in token_list:
            if token1 != token2:
                if token_indices[token1] < token_indices[token2]:
                    token_list[token_list.index(token1)], token_list[token_list.index(token2)] = token_list[token_list.index(token2)], token_list[token_list.index(token1)]
    #print(token_list)
    #print('-----')
    return token_list
         
            
def post_process_token_groups(key_set:dict=None):
    for key in key_set.keys():
        token_list = key_set[key]
        token_dict_indices = {token:i for i, token in enumerate(token_list)}
            
        token_list_sorted = sorted(token_list, key=len, reverse=True)
        #print(token_list_sorted)
        tokens_to_keep=[]
        for token in token_list_sorted:
            if not check_token_presence_within_list(token=token, token_list=tokens_to_keep):
                tokens_to_keep.append(token)
        #print(f'tokens_to_keep {tokens_to_keep}')
        tokens_rearranged = rearrange_token_order(token_key_str= key, token_list=tokens_to_keep, token_indices = token_dict_indices)
            #print(tokens_rearranged)
            #print('----')
        key_set[key] = tokens_rearranged
    return key_set
            
                
def form_token_groups(unified_dict:dict=None, tokens:list=None, bboxes:list = None):
    key_set = {key:[] for key in unified_dict.keys()}
    val_set = {unified_dict[key]:[] for key in unified_dict.keys()}
    #print(val_set)
    #tokens = [token for token in tokens if token not in (' ', '')]
    token_map = {token:[] for token in tokens}
    other_map = {}
        
    for key in key_set.keys():
        for token in tokens:
            token = str(token)
            #key_list = key.split(" ")
            if token in key:        
                if token not in key_set[key]:
                    key_set[key].append(token)
                    token_map[token].append('key')

        ## add found tokens in the respective dicts
    for val in val_set.keys():
        for token in tokens:
            token = str(token)
            if token in str(val):
                if token not in val_set[val]:
                    val_set[val].append(token)
                    token_map[token].append('val')



    key_set = post_process_token_groups(key_set=key_set)
    val_set = post_process_token_groups(key_set=val_set)
        
        
    ## correct the order
        
    for key in key_set.keys():
        token_list = key_set[key]
        token_indices_in_key =[]
        tokens_with_key_index = {}
        for token in token_list:
            token_index = key.find(token)
            token_indices_in_key.append(token_index)
            tokens_with_key_index[token_index] = token
        token_indices_in_key.sort()
        tokens_in_order = [tokens_with_key_index[i] for i in token_indices_in_key]
        key_set[key] = tokens_in_order

    for val in val_set.keys():
        token_list = val_set[val]
        token_indices_in_val =[]
        tokens_with_val_index = {}
        for token in token_list:
            token_index = val.find(token)
            token_indices_in_val.append(token_index)
            tokens_with_val_index[token_index] = token
        token_indices_in_val.sort()
        tokens_in_order = [tokens_with_val_index[i] for i in token_indices_in_val]
        val_set[val] = tokens_in_order
            


    for token_key in token_map.keys():
        token_key = str(token_key)
        if len(token_map[token_key]) == 0:
            #other_map[token_key]
            token_map[token_key].append('other')

    ##post process the token maps by edit distance
    for key in key_set.keys():
        if len(key_set[key]) == 0:
            print(f'key {key}')
            for token in token_map.keys():
                if token_map[token][0] == 'other' and editdistance.eval(key, token) <= 2:
                    print(f'{key}, {token}')
                    token_map[token] = ['key']
                    key_set[key].append(token)

    for val in val_set.keys():
        if len(val_set[val]) == 0:
            #print(f'val {val}')
            for token in token_map.keys():
                #print(f'{val} {token} {token_map[token]} {editdistance.eval(val, token)}')
                if token_map[token][0] == 'other' and editdistance.eval(val, token) <= 2:
                    #print(f'{val}, {token}')
                    token_map[token] = ['val']
                    val_set[val].append(token)

    return key_set, val_set, token_map

    ## Need to store the indices of input ids start and end not the ids themselves

def label_input_ids(key_set:dict=None, val_set:dict=None, input_id_map:dict=None, tokenizer:AutoTokenizer=None, input_ids:list=None):
    label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        
    labels = {}
  
    for _, token_list_key in key_set.items():
        tokenized_tokens = [tokenizer.tokenize(token) for token in token_list_key]
        token_list_map = [input_id_map[tokens[-1]] for tokens in tokenized_tokens]
        for j,id in enumerate(token_list_map):  
            if j ==0:
                labels[id] = label_vals['B-QUESTION']
                            
            if j>0:
                labels[id] = label_vals['I-QUESTION']
        
    for _, token_list_val in val_set.items():
        tokenized_tokens = [tokenizer.tokenize(token) for token in token_list_val]
        token_list_map = [input_id_map[tokens[-1]] for tokens in tokenized_tokens]
        for j,id in enumerate(token_list_map):  
            if j ==0:
                labels[id] = label_vals['B-ANSWER']
                            
            if j>0:
                labels[id] = label_vals['I-ANSWER']
                
    for id in input_ids:
        if id not in labels.keys():
            labels[id] = label_vals['O']
    labels_list = [labels[id] for id in input_ids]
    return labels_list
    
def form_entities(unified_dict:dict=None, input_ids:list= None, input_id_map:dict = None, tokenizer:AutoTokenizer=None, tokens:list = None, bboxes:list = None):
    label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
    label_vals_simplified = {"OTHER": 0, "QUESTION" : 1, "ANSWER" : 2}
    
    entities = {'start':[], 'end':[], 'label':[]}
    token_group_key, token_group_val, token_group = form_token_groups(unified_dict=unified_dict, tokens=tokens, bboxes=bboxes)
    entity_key_index_mapping = {} # top map entity and corresponding key/val appearing in unified dic for later use
    for i, (key, tokens) in enumerate(token_group_key.items()):
        if len(tokens)>0:
            token_first = tokens[0]
            token_last = tokens[-1]
            token_first_tokenized = tokenizer.tokenize(token_first)
            token_last_tokenized = tokenizer.tokenize(token_last)

            token_first_tokenized = [token for token in token_first_tokenized if token not in (' ', '', '\t', '\n','▁') and token not in set(str(string.punctuation))]
            token_last_tokenized = [token for token in token_last_tokenized if token not in (' ', '', '\t', '\n','▁') and token not in set(str(string.punctuation))]
                
            id_list_key_first_token = [input_id_map[t] for t in token_first_tokenized]
            id_list_key_last_token = [input_id_map[t] for t in token_last_tokenized]
                
            entities['start'].append(input_ids.index(id_list_key_first_token[0]))
            entities['end'].append(input_ids.index(id_list_key_last_token[-1]) + 1)
            entities['label'].append(label_vals_simplified['QUESTION'])
            if key == 'förderhöhe':
                print(f'{key} {i}')
            entity_key_index_mapping[i] = key
        
    max_val = max(entity_key_index_mapping.keys()) + 1
        
    for i, (key, tokens) in enumerate(token_group_val.items()):
        if len(tokens) >0:
            token_first = tokens[0]
            token_last = tokens[-1]
            token_first_tokenized = tokenizer.tokenize(token_first)
            token_last_tokenized = tokenizer.tokenize(token_last)

            id_list_key_first_token = [input_id_map[str(t).strip()] for t in token_first_tokenized]
            id_list_key_last_token = [input_id_map[str(t).strip()] for t in token_last_tokenized]
            token_first_tokenized = [token for token in token_first_tokenized if token not in (' ', '', '\t', '\n','▁') and token not in set(str(string.punctuation))]
            token_last_tokenized = [token for token in token_last_tokenized if token not in (' ', '', '\t', '\n','▁') and token not in set(str(string.punctuation))]
                
            entities['start'].append(input_ids.index(id_list_key_first_token[-1]))
            
            entities['end'].append(input_ids.index(id_list_key_last_token[-1]) + 1)
                
            entities['label'].append(label_vals_simplified['ANSWER'])
            update_index = int(i) + max_val
                
            entity_key_index_mapping[update_index] = key 
        
    max_val = max(entity_key_index_mapping.keys()) + 1

    for i, (key, tokens) in enumerate(token_group.items()):
        if len(tokens)>0 and 'other' in tokens:
            token = key
            token_tokenized = tokenizer.tokenize(token)
            token_first_tokenized = [token for token in token_tokenized if token not in (' ', '', '\t', '\n','▁') and token not in string.punctuation]
                
                    
            id_list_key = [input_id_map[t] for t in token_tokenized]
            #if len(ids_present) >0:
            entities['start'].append(input_ids.index(id_list_key[0]))
            entities['end'].append(input_ids.index(id_list_key[-1]) + 1)
            entities['label'].append(label_vals_simplified['OTHER'])
            update_index = int(i) + max_val
            entity_key_index_mapping[update_index] = key
                


        
    entity_key_index_mapping_reverse = {entity_key_index_mapping[key]:key for key in entity_key_index_mapping.keys()}
    return entities, entity_key_index_mapping, entity_key_index_mapping_reverse

def key_val_mapping(entities:dict = None, unified_dict:dict=None, key_set:dict=None, val_set:dict=None):
    key_set_indices = {key:i for i,key in enumerate(unified_dict.keys())}
    val_set_indices = {key_set[key]:key_set_indices[key] for key in key_set_indices }

    return key_set_indices , val_set_indices

def form_relations(entities:dict=None, unified_dict:dict = None, key_set:dict=None, val_set:dict=None, entity_key_index_mapping:dict=None, entity_key_index_mapping_reverse:dict = None):
    relations = {'head':[], 'tail':[]}
    #print(entity_key_index_mapping_reverse)
    for key in key_set.keys():
        key_index = entity_key_index_mapping_reverse[key]
        val = unified_dict[key]
        if val not in entity_key_index_mapping_reverse.keys(): # If val is not detected by ocr both key and vals are ommitted from relations
            continue
        val_index = entity_key_index_mapping_reverse[val]
        relations['head'].append(key_index)
        relations['tail'].append(val_index)

    return relations
    
    '''def label_input_ids(self, unified_dict: dict = None, tokens:list=None, bboxes:list=None, input_ids:list = None, input_id_map:dict = None, tokenizer:AutoTokenizer=None):
        label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        
        labels = {}
  
        for key, val in unified_dict.items():
            
            for i, token in enumerate(tokens):  
                
                if token in key:
                    tokenized_token = tokenizer.tokenize(token)
                    #id_list_key =  tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False)
                    id_list_key = [input_id_map[t] for t in tokenized_token]
                    for j,id in enumerate(id_list_key):
                        
                        if j ==0:
                            labels[id] = label_vals['B-QUESTION']
                            
                        if j>0:
                            labels[id] = label_vals['I-QUESTION']
                               
                        #elif j== len(id_list_key) -1:
                        #    labels[id] = label_vals['I-QUESTION']
                            
                        #else:
                        #    labels[id] = label_vals['I-QUESTION']
                elif token in str(val):
                    tokenized_token = tokenizer.tokenize(token)
                    #id_list_val =  tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False)
                    id_list_val = [input_id_map[t] for t in tokenized_token]
                    for j,id in enumerate(id_list_val):
                        if j ==0:
                            labels[id] = label_vals['B-ANSWER']
                            
                            #entities['start'].append(id)
                        if j>0:
                            labels[id] = label_vals['I-ANSWER']
                                
                        #elif j== len(id_list_key) -1:
                        #    labels[id] = label_vals['I-ANSWER']
                            
                        #else:
                        #   labels[id] = label_vals['I-ANSWER']

                
        for id in input_ids:
            if id not in labels.keys():
                labels[id] = label_vals['O']
        labels_list = [labels[id] for id in input_ids]
        return labels_list
    
        def form_token_groups(self, unified_dict:dict=None, tokens:list=None, bboxes:list = None):
        key_set = {key:[] for key in unified_dict.keys()}
        val_set = {unified_dict[key]:[] for key in unified_dict.keys()}
        #print(val_set)
        #tokens = [token for token in tokens if token not in (' ', '')]
        token_map = {token:[] for token in tokens}
        other_map = {}
        
        for key in key_set.keys():
            for token in tokens:
                token = str(token)
                if token in key:
                    if token not in key_set[key]:
                        key_set[key].append(token)
                        token_map[token].append('key')

        for val in val_set.keys():
            for token in tokens:
                token = str(token)
                if token in str(val):
                    if token not in val_set[val]:
                        val_set[val].append(token)
                        token_map[token].append('val')

        key_set = self.post_process_token_groups(key_set=key_set)
        val_set = self.post_process_token_groups(key_set=val_set)


        for token_key in token_map.keys():
            token_key = str(token_key)
            if len(token_map[token_key]) == 0:
                #other_map[token_key]
                token_map[token_key].append('other')

        return key_set, val_set, token_map'''
        