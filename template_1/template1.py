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

pytesseract.pytesseract.tesseract_cmd = r'C://Program Files//Tesseract-OCR//tesseract.exe'
pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))

header = 'Zugelassene Überwachungsstelle Aufzüge'
file_name = '20181119-32753-1891960176-100-421500.docx'
page_no = 'Seite 1 von 1'


utility_info ={
    'client_address' :  {
  'line1' : 'UniImmo:Deutschland',
  'line2' : 'c/o Union Investment Real Estate GmbH',
  'line3' : 'Valentinskamp 70',
  'line4' : 'D-20355 Hamburg'
},

'client_address_config' : {
  'font' : 'Arial',
  'line_break' : 12,
  'font_size' : 12
},

'evaluator_address' : {
  'address' :{
    'line1' : 'DEKRA Automobil GmbH',
    'line2' : 'Industrie, Bau und Immobilien',
    'line3' : 'Niederlassung Hamburg',
    'line4' : 'Essener Bogen 10',
    'line5' : '22419 Hamburg'
    },
  'Telefon' : '+49.40.23603-0',
  'Telefax' : '+49.40.23603-810',
  'Kontakt' : {
    'line1' : 'Dipl.-Ing. (FH) Olaf Graf',
    'line2' :'Tel. direkt +49.40.23603-839',
    'line3' : 'E-Mail olaf.graf@dekra.com',
  },
  
  'Anlagenschlüssel' : ' AGD669E7NK'
},

'evaluator_address_config' : {
  'font':'Arial',
  'line_break' : 10,
  'font_size' : 8,
  'new_line':['Telefax', 'kontakt']
}
}

global_keys = {

'test_certificate' : {
  'Prüfbescheinigung' : 'Wiederkehrende Prüfung (Hauptprüfung)'
},

'test_certificate_config' : {
  'font_size' : 16,
  'font_type': 'Arial-Bold',
  'line-break' : 25 # should be 25-30
},

'test_certificate_results' : {
  'Prüfgrundlage' : 'gem. § 16 BetrSichV', #Str some code related to legal documents
  'Objektstandort' : 'Bürogebäude, Valentinskamp 70, 20355 Hamburg, Aufzug E', #str address
  'Objektart / Anlage' : 'Personenaufzug', #Str type of elevator
  'Fabrik-Nr.' : '1118365', #long 6-9 digits
  'Arbeitgeber' : utility_info['client_address']['line1'],
  'Eigennummer' : 'HC-ID 13011', # str(2)-str(2) 5 digit number,
  'Eigenname' : 'WE 1172',
  'Verwendete Messgeräte' : 'Profitest 0100, Prüfgewichte', ## Not clear str followed by some 4 digit num and then str 
},

'test_certificate_results_config' : {
  'font_size' : 10,
  'font-type-keys' : 'Arial-Bold',
  'font-type-vals' : 'Arial',
  'vertical-left-only' : ['Objektstandort', 'Arbeitgeber'],
  'line-break' : 20,
  'key-val-spacing': 140
  
},

'technical_specifications' :{
    'Baujahr': 2011, #int year 2000 < year <2023
    'Wartungsfirma': 'Schindler Aufzüge und Fahrtreppen GmbH Region Nord', #Str need to go through the template to decide the pattern
    'Errichtungsgrundlage' : 'EN 81-1: 1998+A3: 2009', # contains alpha numerical, year, need to check more docs to get the patterm
    'Antrieb / Aufhängung' : 'Treibscheibe / 2:1', # contains part name followed by ratio (only key needs to be split by '/' to generate synonyms for each key)
    'Hersteller': 'Schindler Aufzüge und Fahrtreppen GmbH Region Nord',
    'Tragfähigkeit' : '1350 kg / 18 Pers', # kg and persons
    'Haltestellen / Zugangstellen' : '26 / 26', #int
    'Geschwindigkeit': '4,00 m/s', #time / speet
    'Förderhöhe' : '93,68 m' #distance in meters
},

'technical_specifications_config': {
  'font_size' : 10,
  'font-type-keys' : 'Arial-Bold',
  'font-type-vals' : 'Arial',
  'line-break' : 20,
  'key-val-spacing' : 140
    
}}


class Template_Dekra(Template):
    
    def __init__(self, start_x: int = None, start_y:int = None, token_spacing: int= None
                 , line_spacing:int = None, key_spacing:int = None, header_spacing:int = None, section_spacing:int = None, line_break:str=None,  header:str= None, file_name:str= None, report_name:str=None, page_no:int = 1):
        super().__init__()
        self.start_x = start_x
        self.start_x_temp = start_x
        self.start_y = start_y
        self.start_y_temp = start_y
        self.token_spacing = token_spacing
        self.line_spacing = line_spacing
        self.key_spacing = key_spacing
        self.header_spacing = header_spacing
        self.section_spacing = section_spacing
        self.line_break = line_break
        self.header = header
        self.file_name = file_name
        self.report_name = report_name
        
        self.image_name = report_name[:-4]+'.jpg'
        self.page_no = page_no
        
    
    def next_line(self, start_y: int, line_break: int):
        return start_y - line_break
    
    def rearange_key_vals_test_results(self, dict_list : dict, keys_to_keep: list, indices: list): ## keys which needs to appear either of the positions mentioned in indices usually 1,6
        for i , (key, val) in enumerate(dict_list):
            if key in keys_to_keep and i not in indices:
                swap_index = random.choice(indices) if len(indices) > 1 else indices[0] 
                dict_list[i], dict_list[swap_index] = dict_list[swap_index], dict_list[i]
                if len(indices)> 1 :
                    indices.pop(indices.index(swap_index))
    
        return dict_list
    
    def draw_client_address(self, client_address, client_address_config, canvas, x, y):
        canvas.setFont(client_address_config['font'], client_address_config['font_size'])
        keys_sorted = [key for key in client_address.keys()]
        
        for key in keys_sorted:
            canvas.drawString(x, y, client_address[key])
            y = self.next_line(y, client_address_config['line_break'])
        return x, y
    
    def draw_evaluator_address(self, evaluator_address, evaluator_address_config, canvas, x, y, y_temp):
        canvas.setFont(evaluator_address_config['font'], evaluator_address_config['font_size'])
        start_x_temp = random.choice(np.arange(400,430, 10))
        y = y_temp
        count = 0
        for key in evaluator_address.keys():
            if not type(evaluator_address[key]) == dict: 
                canvas.drawString(start_x_temp, y, key + ' : ' + evaluator_address[key])
                count += 1
                y = self.next_line(y, evaluator_address_config['line_break'])
            else:
                for k in evaluator_address[key].keys():
                    canvas.drawString(start_x_temp, y, evaluator_address[key][k])
                    count += 1
                    y = self.next_line(y, evaluator_address_config['line_break'])
                if count%5==0 or count%3==0: # Adding random new lines
                    y = self.next_line(y, evaluator_address_config['line_break'])
        return x, y
    
    def shuffle_dict(self, dict_):
        items_test_certificate_results = list(dict_.items())
        random.shuffle(items_test_certificate_results)
        return items_test_certificate_results
    

    
    def draw_test_certificate_results(self, test_certificate_results, test_certificate_results_config, canvas, x, y):
        items_test_certificate_results = self.shuffle_dict(test_certificate_results)
        items_test_certificate_results = self.rearange_key_vals_test_results(items_test_certificate_results, test_certificate_results_config['vertical-left-only'], [1,6])
        start_x_temp = x
        new_lines = []
        for i , (key, val) in enumerate(items_test_certificate_results):
            if i == 5 and key not in test_certificate_results_config['vertical-left-only']:
                start_x_temp = start_x_temp + 2 * test_certificate_results_config['key-val-spacing']
                y = new_lines[-4]
            canvas.setFont(test_certificate_results_config['font-type-keys'], test_certificate_results_config['font_size'])
            canvas.drawString(start_x_temp, y , key)
            #start_x_temp = start_x_temp + test_certificate_results_config['key-val-spacing']
            canvas.setFont(test_certificate_results_config['font-type-vals'], test_certificate_results_config['font_size'])
            canvas.drawString(start_x_temp + test_certificate_results_config['key-val-spacing'], y, val)
            y = self.next_line(y, test_certificate_results_config['line-break'])
            new_lines.append(y)

        return x, new_lines[-1]
    
    def unify_keys_vals(self, dict_):
        unified_dict = {}
        for key_1 in dict_.keys():
            if 'config' not in key_1:
                for key_2 in dict_[key_1].keys():
                    unified_dict[key_2] = dict_[key_1][key_2]
        return unified_dict
    
    def get_ocr_data(self, conf_val:float=50, image_path:str= 'form.jpg'):
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
    
    def add_tokens_tokenizer(self, tokens:list=None, bboxes:list=None, tokenizer:AutoTokenizer=None, model:LayoutLMv2ForRelationExtraction=None):
        
        tokens, _ = self.preprocess_tokens(tokens=tokens, bboxes=bboxes)
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
        

    def preprocess_tokens(self,tokens:list=None, bboxes:list=None):
        speical_characters = set(string.punctuation)
        tokens_preprocessed = []
        bboxes_preprocessed = []
        
        for i, token in enumerate(tokens):
            token = token.lower()
            if token not in speical_characters and token not in (' ',  ''):
                for char_ in speical_characters:
                    token = token.strip(char_)
                tokens_preprocessed.append(token)
                bboxes_preprocessed.append(bboxes[i])
        
        return tokens_preprocessed, bboxes_preprocessed

    def encode_tokens(self, tokens:list=None, bboxes:list = None, tokenizer : AutoTokenizer = None) :
        input_ids = []
        bboxes_tokenized = []
        #tokens, bboxes = self.preprocess_tokens(tokens=tokens, bboxes=bboxes)
        for i in range(len(tokens)):
            tokenized_token = tokenizer.tokenize(tokens[i])
            #print(tokenized_token)
            #print(bboxes[i])
            for input_id in tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False):
                input_ids.append(input_id)
                bboxes_tokenized.append(bboxes[i]) 
        return input_ids, bboxes_tokenized

    def label_input_ids(self, unified_dict: dict = None, tokens:list=None, bboxes:list=None,input_ids:list = None, tokenizer:AutoTokenizer=None):
        label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        #input_ids = list(set(input_ids))
        
        #input_id_token = {tokenizer.decode(id):id for id in input_ids}
        input_id_token = {id:tokenizer.decode(id) for id in input_ids}
        #print(input_id_token) 
        #tokenizer_2 = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        #model_2 = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
        labels = {}
        for key, val in unified_dict.items():
            #id_list_key = []
            #id_list_val = []
            #key = str(key).lower()
            #val = str(val).lower()
            #key_list = [token for token in key.split(' ')]
            #val_list = [token for token in val.split(' ')]
            #print([val for val in val_list if 'uni' in val])
            #for char in string.punctuation:
            #    key_list = [key.strip(char) for key in key_list]
            #    val_list = [val.strip(char) for val in val_list]
            
            for i, token in enumerate(tokens):
            #if 'uni' in val.lower():
            #    print(val)
            #key = str(key).lower()
            #val = str(val).lower()
            #key_list = [token for token in key.split(' ')]
            #val_list = [token for token in val.split(' ')]
            #print([val for val in val_list if 'uni' in val])
            #for char in string.punctuation:
            #    key_list = [key.strip(char) for key in key_list]
            #    val_list = [val.strip(char) for val in val_list]
            
            #print([val for val in val_list if 'uni' in val])
            #print(key_list)
            #print(val_list)
            #print('-----')
            
            #id_list_key = [input_id_token[token] for token in key_list if token in input_id_token.keys()]
            #id_list_val = [input_id_token[token] for token in val_list if token in input_id_token.keys()]
                
                if token in key:
                    tokenized_token = tokenizer.tokenize(token)
                    #print(tokenized_token)
                    #print(bboxes[i])
                    id_list_key =  tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False)
                    #id_list_key = [input_id_token[t] for tokenized_token in tokenizer.tokenize(token) for t in tokenized_token if t in input_id_token.keys()]
                    
                    for i,id in enumerate(id_list_key):
                    #token, i = element[0], element[1]
                    #id = input_id_token[token]
                        if i ==0:
                            labels[id] = label_vals['B-QUESTION']
                        else:
                            labels[id] = label_vals['I-QUESTION']
                elif token in str(val):
                    tokenized_token = tokenizer.tokenize(token)
                    #print(tokenized_token)
                    #print(bboxes[i])
                    id_list_val =  tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False)
                    for i,id in enumerate(id_list_val):
                    #token, i = element[0], element[1]
                    #id = input_id_token[token]
                        if i ==0:
                            labels[id] = label_vals['B-ANSWER']
                        else:
                            labels[id] = label_vals['I-ANSWER']

                #else:
                #    tokenized_token = tokenizer.tokenize(token)
                #    #print(tokenized_token)
                #    #print(bboxes[i])
                #    id_list_val =  tokenizer.encode(text=tokenized_token, boxes = [bboxes[i] for j in range(len(tokenized_token))], is_pretokenized=True, add_special_tokens=False)
                #    for id in enumerate(id_list_val):
                #    #token, i = element[0], element[1]
                #    #id = input_id_token[token]
                #        labels[id] = label_vals['O']
                        
                
            
                #print(id_list_key)
                
                    
        #print(len(labels))
        for id in input_ids:
            if id not in labels.keys():
                labels[id] = label_vals['O']
        #print(len(labels))
        #print(len(input_ids))
        labels_list = [labels[id] for id in input_ids]
        #labels_list = [labels[input_id_token[token]] for token in tokens if token in input_id_token.keys()]
        return labels_list
    
    def form_entities(self, unified_dict:dict=None, input_ids:list= None, tokenizer:AutoTokenizer=None, tokens:list = None):
        label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        label_vals_simplified = {"OTHER": 0, "QUESTION" : 1, "ANSWER" : 2}
        #label_vals_simplified = {0: "HEADER", 1: "QUESTION", 2:"ANSWER"}
        
        #input_ids = list(set(input_ids))
        input_id_token = {tokenizer.decode(id):id for id in input_ids}
        entities = {'start':[], 'end':[], 'label':[]}
        for key, val in unified_dict.items():
            #if 'uni' in val.lower():
            #    print(val)
            key = str(key).lower()
            val = str(val).lower()
            key_list = [token for token in key.split(' ')]
            val_list = [token for token in val.split(' ')]
            #print([val for val in val_list if 'uni' in val])
            for char in string.punctuation:
                key_list = [key.strip(char) for key in key_list]
                val_list = [val.strip(char) for val in val_list]
            
            #print([val for val in val_list if 'uni' in val])
            #print(key_list)
            #print(val_list)
            #print('-----')
            id_list_key = [input_id_token[token] for token in key_list if token in input_id_token.keys()]
            id_list_val = [input_id_token[token] for token in val_list if token in input_id_token.keys()]

            ids_others = set(input_ids)
            ids_key_val = []
            for i,id in enumerate(id_list_key):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                ids_key_val.append(id)
                if i ==0:
                    entities['start'].append(id)
                elif i == len(id_list_key) - 1:
                    entities['end'].append(id)
                    entities['label'].append(label_vals_simplified["QUESTION"])
                
                    
            for i,id in enumerate(id_list_val):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                ids_key_val.append(id)
                if i ==0:
                    entities['start'].append(id)
                elif i == len(id_list_key) - 1:
                    entities['end'].append(id)
                    entities['label'].append(label_vals_simplified["ANSWER"])
            
            #print(ids_others)
            #print(set(ids_key_val))

        ids_others = ids_others - set(ids_key_val)
        #print(f'ids_others {ids_others}')
        ids_others =  list([int(id) for id in ids_others])
        #print(f'ids_others {ids_others}')
        entities['start'].append(min(ids_others))
        entities['end'].append(max(ids_others))
        entities['label'].append(label_vals_simplified['OTHER'])

        return entities

    '''def encode_new_tokens(self,tokens:list=None, bboxes:list=None, tokenizer:AutoTokenizer=None):
        #tokens, bboxes = self.preprocess_tokens(tokens, bboxes)
        input_ids = tokenizer.encode(text = tokens, boxes = bboxes, is_pretokenized=False) 
        return input_ids'''
    
    
    '''def form_entities(self, unified_dict:dict=None, input_ids:list= None, tokenizer:AutoTokenizer=None, tokens:list = None):
        label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        label_vals_simplified = {"OTHER": 0, "QUESTION" : 1, "ANSWER" : 2}
        #label_vals_simplified = {0: "HEADER", 1: "QUESTION", 2:"ANSWER"}
        
        input_ids = list(set(input_ids))
        input_id_token = {tokenizer.decode(id):id for id in input_ids}
        entities = {'start':[], 'end':[], 'label':[]}
        for key, val in unified_dict.items():
            #if 'uni' in val.lower():
            #    print(val)
            key = str(key).lower()
            val = str(val).lower()
            key_list = [token for token in key.split(' ')]
            val_list = [token for token in val.split(' ')]
            #print([val for val in val_list if 'uni' in val])
            for char in string.punctuation:
                key_list = [key.strip(char) for key in key_list]
                val_list = [val.strip(char) for val in val_list]
            
            #print([val for val in val_list if 'uni' in val])
            #print(key_list)
            #print(val_list)
            #print('-----')
            id_list_key = [input_id_token[token] for token in key_list if token in input_id_token.keys()]
            id_list_val = [input_id_token[token] for token in val_list if token in input_id_token.keys()]

            ids_others = set(input_ids)
            ids_key_val = []
            for i,id in enumerate(id_list_key):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                ids_key_val.append(id)
                if i ==0:
                    entities['start'].append(id)
                elif i == len(id_list_key) - 1:
                    entities['end'].append(id)
                    entities['label'].append(label_vals_simplified["QUESTION"])
                
                    
            for i,id in enumerate(id_list_val):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                ids_key_val.append(id)
                if i ==0:
                    entities['start'].append(id)
                elif i == len(id_list_key) - 1:
                    entities['end'].append(id)
                    entities['label'].append(label_vals_simplified["ANSWER"])
            
            #print(ids_others)
            #print(set(ids_key_val))

        ids_others = ids_others - set(ids_key_val)
        #print(f'ids_others {ids_others}')
        ids_others =  list([int(id) for id in ids_others])
        #print(f'ids_others {ids_others}')
        entities['start'].append(min(ids_others))
        entities['end'].append(max(ids_others))
        entities['label'].append(label_vals_simplified['OTHER'])

        return entities'''
            

    '''def label_indices(self, unified_dict:dict=None, input_ids:list=None, tokenizer:AutoTokenizer=None, tokens :list=None ):
        label_vals = {'O' : 0, 'B-QUESTION' : 1, 'B-ANSWER' : 2, 'B-HEADER' : 3, 'I-ANSWER' : 4, 'I-QUESTION' : 5, 'I-HEADER' : 6}
        input_ids = list(set(input_ids))
        input_id_token = {tokenizer.decode(id):id for id in input_ids}
        #input_id_token = {id:tokenizer.decode(id) for id in input_ids}
        #print(input_id_token)
        #tokenizer_2 = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        #model_2 = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
        labels = {}
        for key, val in unified_dict.items():
            #if 'uni' in val.lower():
            #    print(val)
            key = str(key).lower()
            val = str(val).lower()
            key_list = [token for token in key.split(' ')]
            val_list = [token for token in val.split(' ')]
            #print([val for val in val_list if 'uni' in val])
            for char in string.punctuation:
                key_list = [key.strip(char) for key in key_list]
                val_list = [val.strip(char) for val in val_list]
            
            #print([val for val in val_list if 'uni' in val])
            #print(key_list)
            #print(val_list)
            #print('-----')
            id_list_key = [input_id_token[token] for token in key_list if token in input_id_token.keys()]
            id_list_val = [input_id_token[token] for token in val_list if token in input_id_token.keys()]
            #id_list_key = [(token, key.index(token)) for token in input_id_token.keys() if token in key]
            #id_list_val = [(token, key.index(token)) for token in input_id_token.keys() if token in val]
            
            
            #print(id_list_key)
            for i,id in enumerate(id_list_key):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                if i ==0:
                    labels[id] = label_vals['B-QUESTION']
                else:
                    labels[id] = label_vals['I-QUESTION']
                    
            for i,id in enumerate(id_list_val):
                #token, i = element[0], element[1]
                #id = input_id_token[token]
                if i ==0:
                    labels[id] = label_vals['B-ANSWER']
                else:
                    labels[id] = label_vals['I-ANSWER']
                    
        for id in input_ids:
            if id not in labels.keys():
                labels[id] = label_vals['O']
        
        #labels_list = [labels[id] for id in input_ids]
        labels_list = [labels[input_id_token[token]] for token in tokens if token in input_id_token.keys()]
        return labels_list'''
                    
                
            
                    
            
            
    
    def draw_technical_specifications(self, technical_specifications, technical_specifications_config, canvas, x, y):
        start_x_temp = x
        start_y_temp = y
        shuffled_dict_tuple = self.shuffle_dict(technical_specifications)
        for i, (key, val) in enumerate(shuffled_dict_tuple):
            if i == 4:
                start_x_temp += 2*technical_specifications_config['key-val-spacing']
                start_y_temp = y
            
            canvas.setFont(technical_specifications_config['font-type-keys'], technical_specifications_config['font_size'])
            canvas.drawString(start_x_temp, start_y_temp, str(key))
            canvas.setFont(technical_specifications_config['font-type-vals'], technical_specifications_config['font_size'])
            #start_x_temp += technical_specifications_config['key-val-spacing']
            if len(str(val).split(" ")) > 3:
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, ' '.join(str(val).split(" ")[:3]))
                start_y_temp = self.next_line(start_y_temp, 10)
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, ' '.join(str(val).split(" ")[3:]))
            else:
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, str(val))
            start_y_temp = self.next_line(start_y_temp, technical_specifications_config['line-break'])
            
        return x, start_y_temp

    def draw_report(self,header:dict=None, report_name:str='form.pdf'):
  
        c = Canvas(report_name)
        canvas = c
        canvas.setPageSize(letter)
        canvas.setLineWidth(.3)
        canvas.setFont('Arial-Bold', 10) 
  
        canvas.setFillColor(HexColor(0x000000))
        canvas.drawString(self.start_x,self.start_y, header)  
        canvas.setFont('Arial', 9)
  
        new_line = self.next_line(self.start_y, self.line_break)
  
        canvas.drawString(self.start_x,new_line, file_name)
  
        new_line = self.next_line(new_line, 4)
        canvas.line(self.start_x, new_line, 600, new_line)
  
        ## Section Spacing
        new_line = self.next_line(new_line, self.section_spacing)
  
        ## Used for evaluator address
        new_line_temp = new_line + 10
  
        # Section 1
        ## Client Address

        _, new_line = self.draw_client_address(utility_info['client_address'], utility_info['client_address_config'], canvas, self.start_x, new_line)

        #canvas.line(480,747,580,747)
  
        #Section 2
        _, new_line = self.draw_evaluator_address(utility_info['evaluator_address'], utility_info['evaluator_address_config'], canvas, self.start_x, new_line, new_line_temp)
  
        ## Section 3 test_certificate
        new_line = self.next_line(new_line, self.section_spacing)
        canvas.setFont(global_keys['test_certificate_config']['font_type'], global_keys['test_certificate_config']['font_size'])
        canvas.drawString(self.start_x, new_line, list(global_keys['test_certificate'].keys())[0])
        new_line = self.next_line(new_line, global_keys['test_certificate_config']['line-break'])
        canvas.setFont(global_keys['test_certificate_config']['font_type'], global_keys['test_certificate_config']['font_size'] - 2)
        canvas.drawString(self.start_x, new_line, global_keys['test_certificate']['Prüfbescheinigung'])
  
        ## Section 4 test_certificate_results
        new_line = self.next_line(new_line, self.section_spacing)

        ## Section 5 
        _, new_line = self.draw_test_certificate_results(global_keys['test_certificate_results'], global_keys['test_certificate_results_config'], canvas, self.start_x, new_line)
        new_line = self.next_line(new_line, self.line_break)
        
        ## Section 6
        _, new_line = self.draw_technical_specifications(global_keys['technical_specifications'], global_keys['technical_specifications_config'], canvas, self.start_x, new_line)
        #_, new_line = self.draw_test_certificate_results(test_certificate_results, test_certificate_results_config, canvas, self.start_x, new_line)

        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(f'{report_name[:-4]}.jpg', 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        
if __name__=='__main__':
    start_x = 30
    start_x_temp = start_x
    start_y = 750
    start_y_temp = start_y
    token_spacing = 100
    line_spacing = 8
    count = 0
    key_spacing = 200
    header_spacing = 15
    section_spacing = 30
    line_break = 20

    #file_name='form.pdf'
    header = 'Zugelassene Überwachungsstelle Aufzüge'
    file_name = '20181119-32753-1891960176-100-421500.docx'
    report_name = 'form.pdf'
    page_no = 'Seite 1 von 1'

    template1 = Template_Dekra(start_x = start_x,
        start_y = start_y,
        token_spacing = token_spacing,
        line_spacing = line_spacing,
        key_spacing = key_spacing,
        header_spacing = header_spacing,
        section_spacing = section_spacing,
        line_break = line_break,
        header = header,
        file_name = file_name,
        report_name=report_name,
        page_no = page_no)
    #draw_report(header=header, report_name='form.pdf')

    #template1.draw_report(header=header, report_name='form.pdf')
    #pages = convert_from_path('form.pdf', 500)
    #pages[0].save(f'form.jpg', 'JPEG')
    model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    #print(f'len(tokenizer) {len(tokenizer)}')
    tokens, bboxes, image = template1.get_ocr_data(image_path='form.jpg')
    #print([token for token in tokens if token in set(string.punctuation) and token in (' ', '')])
    #print('------------------------')
    #template1.add_tokens(tokens=tokens)
    #tokens = [token.lower() for token in tokens if token != ' ']
    #tokens, bboxes = template1.preprocess_tokens(tokens=tokens, bboxes=bboxes)

    #tokens = np.load('tokens_temp.npy')
    #bboxes = np.load('bboxes_temp.npy')
    #print(f'len(tokens) {len(tokens)}')
    #print(f'len(bboxes) {len(bboxes)}')
    #plt.figure(figsize=(20,10))
    #plt.imshow(image)
    #tokenizer, model = template1.add_tokens_tokenizer(tokens = tokens, tokenizer = tokenizer, model = model)
    input_ids, bboxes = template1.encode_tokens(tokens=tokens, bboxes=bboxes, tokenizer=tokenizer)
    #tokens, bboxes = template1.preprocess_tokens(tokens= tokens, bboxes= bboxes)
    #print(tokens)
    #print(f'len of tokens {len(tokens)}')
    #time.sleep(10)

    #print(input_ids)
    #tokenizer = AutoTokenizer.from_pretrained('layout_xlm_base_tokenizer')
    #tokenizer = AutoTokenizer.from_pretrained('layout_xlm_base_tokenizer_alt')
    #print(f'len(tokenizer) {len(tokenizer)}')
    
    #input_ids = tokenizer.encode(text = tokens, boxes = bboxes, is_pretokenized=False)  
    key_vals_unified = template1.unify_keys_vals(global_keys)
    labels = template1.label_input_ids(unified_dict=key_vals_unified,tokens = tokens,  bboxes = bboxes, input_ids=input_ids, tokenizer=tokenizer)
    print(len(input_ids))
    print(len(labels))
    #for id, label in zip(input_ids, labels):
    #    print(f'{tokenizer.decode(id)}, {label}')
    
    #print(tokenizer.tokenize('Prüfbescheinigung, Wiederkehrende Prüfung (Hauptprüfung)'.lower()))
    #for id in input_ids:
    #    print(f'{tokenizer.decode(id)}')
    #print(f' outside method lenght of input ids {len(input_ids)}') 
    #for id in input_ids:
    #    print(tokenizer.decode([id]))