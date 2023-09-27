import os


from Template import Template

import json
#pwd = os.getcwd()
#os.chdir("..")
import sys
sys.path.append('..')

from utility_functions.utilities_kie import *
from utility_functions.utilitites_templates import * 
import pandas as pd
from reportlab.lib import colors
from reportlab.graphics.shapes import *

#os.chdir(pwd)
from faker import Faker
locales = ['de_DE']
Faker.seed(0)


key_mappings = {
"evaluator_addres_object":["Anschrift des Bewerters", "Tel","Fax"],
"adkb_object":["Arbeitsschein-Nr","Datum","Kunden-Nr","Ihre Bestellnummer"],
"anst_object":["Anlagen-Nr", "Standort"],
"afst_object":["Auftrags-Nr", "Service Techniker"],
"ankunft_object":["Ankunft vor Ort", "Tätigkeit"],
"info_techniker_obj":["Information Servicetechniker"]
}

rect_color_map = {"lightgrey":colors.lightgrey, "white":colors.white}

class Template_Schindler(Template):
    def __init__(self, start_x:int, start_y:int, shuffle_dict:bool=None, rect_colors:list=None, sentance_gen_type:str=None, keys_to_include:list=None, file_name:str= None, fonts_dir:str=None, sentances_dir:str=None):
        self.start_x = start_x
        self.start_y = random.choice((start_y[0], start_y[1]))
        self.shuffle_dict=shuffle_dict
        self.rect_colors = rect_colors
        self.sentance_gen_type = sentance_gen_type
        self.rect_color = rect_color_map[random.choice(self.rect_colors)]
        self.file_name = file_name
        self.fonts_dir = fonts_dir
        self.sentances_dir = sentances_dir
        self.keys_to_include = keys_to_include
        #print(self.keys_to_include)
        self.fake = Faker(locales)
        with open(f'{self.sentances_dir}/fixed_sentances.json', 'r') as f:
            self.fixed_sentances = json.load(f)

    def init_global_keys(self):
        global_key_vals_no_keys = {
            "Anschrift des Bewerters" : "Schindler Aufzüge und Fahrtreppen GmbH Postfach 63 04 62 22314 Hamburg",
            "Adresse des Besitzers" : "Union Investment Real Estate GmbH DIFA-Fonds 3 c/o STRABAG Property and Facility Services GmbH - Objektbuchhaltung - Postfach 50 10 60 70340 Stuttgart",
            "Ankunft vor Ort" : "04.03.2013",
            "Tätigkeit" : "Wartung",
            "Arbeitsdauer" : "01h00",
            "Total":"01h00"
        }

        global_key_vals_keys = {
            "Tel":"040/53901",
            "Fax":"040/53901 333",
            "ArbeiTätigkeittsschein-Nr": "5454261",
            "Datum" : "05.03.2013",
            "Kunden-Nr": "1442968",
            "Ihre Bestellnummer": "1234",
            "Anlagen-Nr" : "18000088938 /88938",
            "Standort" : "Großer Heegbarg 16, D-22391 Hamburg",
            "Auftrags-Nr": "1804970090",
            "Service Techniker": "Dirk Buscher",
            "Information Servicetechniker":"Intensive Reinigung der Schachtgrube. ##Haufenweise Fliesen lagen in der Schachtgrube.##Reinigen der RAS Anlage. ##Sodexo Schadensnummer. 2017-1-0149"
        }
        return global_key_vals_keys, global_key_vals_no_keys

        
    def generate_tel_fax(self):
        sequence = np.arange(0,9)
        area_code = ''.join([str(random.choice(sequence)) for _ in range(3)])
        tel = area_code + '/'+ ''.join([str(random.choice(sequence[1:])) for _ in range(5)])
        fax = tel +' '+ ''.join([str(random.choice(sequence[1:])) for _ in range(3)])
        return tel, fax
    
    def generate_ranom_sentance(self, token_len:int = 20):
        return self.fake.sentence(nb_words=token_len).replace('\n', ' ')
    
    def genrate_arbeitsschein_nr(self):
        sequence = np.arange(0,9)
        arbeitschein_nr = ''.join([str(random.choice(sequence[1:])) for _ in range(np.random.choice([6,7,8]))])
        return arbeitschein_nr
    
    def generate_kunden_nr(self):
        sequence = np.arange(0,9)
        kunden_nr = ''.join([str(random.choice(sequence[1:])) for _ in range(np.random.choice([7,8]))])
        return kunden_nr
    
    def generate_anlagenummer(self):
        sequence = np.arange(0,9)
        anlagen_nummer_part1 = str(random.choice(sequence[1:]))+''.join([str(random.choice(sequence)) for _ in range(10)])
        anlagen_nummer_part2 = str(random.choice(sequence[1:]))+''.join([str(random.choice(sequence)) for _ in range(4)])
        return f'{anlagen_nummer_part1} /{anlagen_nummer_part2}'
    
    def generate_auftrags_nr(self):
        sequence = np.arange(0,9)
        auftrags_nummer = str(random.choice(sequence[1:]))+''.join([str(random.choice(sequence)) for _ in range(10)])
        return auftrags_nummer
    
    def generate_random_num(self, digits:int=None):
        sequence = np.arange(0,9)
        auftrags_nummer = str(random.choice(sequence[1:]))+''.join([str(random.choice(sequence)) for _ in range(digits)])
        return auftrags_nummer
    
    def populate_keys_key(self, keys_key:dict):
        tel, fax = self.generate_tel_fax()
        keys_key["Tel"] =tel
        keys_key["Fax"] =fax
        keys_key["Arbeitsschein-Nr"] = self.genrate_arbeitsschein_nr()
        keys_key["Kunden-Nr"] = self.generate_random_num(digits=random.choice([7,8]))
        keys_key["Datum"] = self.fake.date(pattern="%d.%m.%Y")
        keys_key["Anlagen-Nr"] = self.generate_anlagenummer()
        keys_key["Auftrags-Nr"] = self.generate_random_num(digits=10)
        keys_key["Standort"] = self.fake.address().replace('\n', ' ')
        keys_key["Service Techniker"] = f'{self.fake.first_name()} {self.fake.first_name()}'.replace('\n', ' ')
        keys_key["Information Servicetechniker"] = self.generate_ranom_sentance(token_len = 20)
        keys_key["Ihre Bestellnummer"] = self.generate_random_num(digits=8)
        #print(keys_key["Information Servicetechniker"])
        
        return keys_key
    
    def filter_keys(self, global_keys_key:dict, global_keys_nokey:dict = None):
        filtered_dicts = {}
        for key in self.keys_to_include:
            if key in global_keys_key:
                filtered_dicts[key] = global_keys_key[key]
            elif key in global_keys_nokey:
                filtered_dicts[key] = global_keys_nokey[key]
        return filtered_dicts
    
    def populate_keys_nokey(self, keys_nokey:dict):
        
        keys_nokey["Anschrift des Bewerters"] = self.fake.company().replace('\n', ' ') +' '+self.fake.address().replace('\n', ' ')
        keys_nokey["Adresse des Besitzers"] = self.fake.company().replace('\n', ' ') +' ' +self.fake.address().replace('\n', ' ')
        keys_nokey["Ankunft vor Ort"] = self.fake.date(pattern="%d.%m.%Y") if random.choice([1,2,3,4]) == 1 else f'{self.fake.date(pattern="%d.%m.%Y")} {self.fake.time()}'
        keys_nokey["Ankunft vor Ort"] = str(keys_nokey["Ankunft vor Ort"]).replace('\n', ' ')
        return keys_nokey
    
    def draw_Address(self,canvas:object=None,x:int=None, y:int=None, address:str=None, font_size:int=None, line_break:int= None ):
        start_x = x
        start_y = y
        canvas.setFont(self.font, font_size)
        
        lines = break_string_recursively(address, random.choice(np.arange(28,33)))
        for line in lines:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, line_break)
        
        return start_x, start_y
    
    def draw_ADKB(self, canvas:object=None, x:int=None, y:int=None, adkb_object:dict=None, font_size:int=None): ## ADKB stands for Arbeitschein_nr, Datum, Kunden-Nr, Bestellnummer
        start_x = x
        start_y = y
        canvas.setFont(self.font, font_size)        
        shuffled_dict = adkb_object
        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(adkb_object)
        else:
            shuffled_dict = [(key, val) for key,val in adkb_object.items()]
        
        for i, (key,val) in enumerate(shuffled_dict):
            if i==0:
                canvas.drawString(start_x, start_y,f'{key}: {val}')
            elif i==1:
                start_x_temp=start_x
                start_x_temp += 100
                canvas.drawString(start_x_temp, start_y,f'/ {key}: {val}')
                start_y = next_line(start_y, 13)
            else:
                canvas.drawString(start_x, start_y,f'{key}: {val}')
                start_y = next_line(start_y, 13)
        
        return start_x, start_y
    
    def draw_rect(self, canvas:object=None, x:int=None, y:int=None, rect_height:int=None, rect_width:int=None):
        start_x = x
        start_y = y
        canvas.setFillColor(self.rect_color)
        canvas.rect(start_x, start_y, rect_width, rect_height,stroke=0, fill=1)
        canvas.line(start_x, start_y, start_x+rect_width, start_y) #x1,y1,x2,y2
        
        canvas.setFillColor(colors.black)
        #canvas.setFont(self.font_bold, font_size)
        #start_y_temp = start_y + rect_height //4
        #canvas.drawString(start_x, start_y_temp, text)
        
        return start_x, start_y

    def draw_anst(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, anst_object:dict=None, key_val_spacing:int=None, line_break:int=None): ## ANST stands for anlage , standord
        start_x = x
        start_y = y
        
        rect_height = 12
        #print(f'line_break {line_break}')
        start_x, start_y = self.draw_rect(canvas=canvas, x=start_x, y=start_y, rect_width=500, rect_height=12)
        canvas.setFont(self.font_bold, font_size)
        
        start_y_temp = start_y + rect_height //4
        text = self.fixed_sentances["information_anlage"]
        canvas.drawString(start_x, start_y_temp, text)
        
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        if self.shuffle_dict:
            anst_object = shuffle_dict(anst_object)
        
        for key,val in anst_object.items():
            canvas.drawString(start_x, start_y, f'{key} :')
            start_x_temp = start_x
            start_x_temp += key_val_spacing
            canvas.drawString(start_x_temp, start_y, val)
            start_y = next_line(start_y, 15)
        
        return start_x, start_y

    def draw_afst(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, afst_object:dict=None, key_val_spacing:int=None, line_break:int=None):
        start_x = x
        start_y = y
        
        canvas.setFont(self.font_bold, font_size)
        rect_height = 12
        #print(f'line_break {line_break}')
        start_x, start_y = self.draw_rect(canvas=canvas, x=start_x, y=start_y, rect_width=500, rect_height=12)
        start_y_temp = start_y + rect_height //4
        text_1 = self.fixed_sentances["information_anlage"]
        text_2 = self.fixed_sentances["wartung"]
        
        canvas.drawString(start_x, start_y_temp, text_1)
        start_x_temp = start_x+key_val_spacing
        canvas.drawString(start_x_temp, start_y_temp, text_2)
        
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        if self.shuffle_dict:
            afst_object = shuffle_dict(afst_object)
        
        for key,val in afst_object.items():
            canvas.drawString(start_x, start_y, f'{key} :')
            start_x_temp = start_x
            start_x_temp += key_val_spacing
            canvas.drawString(start_x_temp, start_y, val)
            start_y = next_line(start_y, 15)
        
        return start_x, start_y
    
    def draw_info_technicker_rect(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, info_techniker_obj:dict=None, rect_width:int=None, rect_height:int=None, line_break:int=None):
        start_x = x
        start_y = y
        canvas.setFont(self.font_bold, font_size)
        
        #print(info_techniker_obj)
        ## Getting key and value and drawing the key
        canvas.setFillColor(colors.white)
        canvas.rect(start_x - 5, start_y + 10, rect_width, -1*rect_height,stroke=1, fill=1)
        canvas.setFillColor(colors.black)
        key = list(info_techniker_obj.keys())[0]
        canvas.drawString(start_x, start_y, f'{key}:')
        val = info_techniker_obj[key]
        start_y = next_line(start_y, line_break)
        lines = break_string_recursively(val, 100)
        #print(key)
        ## Drawing the value
        canvas.setFont(self.font, font_size)
        ys=[]
        for line in lines:
            canvas.drawString(start_x, start_y, line)
            ys.append(start_y)
            start_y = next_line(start_y, line_break)
        
        
        
        
        ## Drawing the rectangle
        
        return start_x, y - rect_height
    
    def draw_ankunft_ort(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, key_val_spacing:int=None, ankunft_object:dict=None, line_break:int = None):
        start_x = x
        start_y = y
        
        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(ankunft_object)
        else:
            shuffled_dict = [(key,val) for key,val in ankunft_object.items()]
        rect_height = 12
        rect_width = 500
        #print(f'line_break {line_break}')
        start_x, start_y = self.draw_rect(canvas=canvas, x=start_x, y=start_y, rect_width=rect_width, rect_height=rect_height)
        canvas.setFont(self.font_bold, font_size)
        
        start_y_temp = start_y + rect_height //4
        keys = [item[0] for item in shuffled_dict]
        
        text_1 = keys[0]
        text_2 = keys[1]
        
        canvas.drawString(start_x, start_y_temp, text_1)
        start_x_temp = start_x+key_val_spacing
        canvas.drawString(start_x_temp, start_y_temp, text_2)
        
        
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        canvas.drawString(start_x, start_y, ankunft_object[keys[0]])
        canvas.drawString(start_x_temp, start_y, ankunft_object[keys[1]])
        start_y = next_line(start_y, line_break)
        
        canvas.line(start_x, start_y, start_x+rect_width, start_y)
        
        return start_x, start_y
    
    def draw_auftrag_sentance(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, sentance:str=None):
        start_x = x
        start_y = y
        canvas.setFont(self.font, font_size)
        
        lines = break_string_recursively(sentance, 100)
        for line in lines:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, random.choice(np.arange(12,15)))
        
        return start_x, start_y 
  
    def init_object(self, global_keys_key:dict=None, global_keys_no_key:dict=None, keys:list=None):
        obj={}
        for key in keys:
            if key in global_keys_key.keys():
                obj[key] = global_keys_key[key]
            elif key in global_keys_no_key.keys():
                obj[key] = global_keys_no_key[key]
        return obj
    
    def draw_footer(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, font_size:int=None, rarity:int=3):
        
        start_x = x
        start_y = y
        
        if random.choice(np.arange(rarity)) == 1:
            canvas.setFont(self.font_bold, font_size)
            sentance = self.fixed_sentances["signature_note"] if self.sentance_gen_type == 'fixed' else self.fake.sentence(nb_words=8)
            canvas.drawString(start_x, start_y, sentance)
        
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        sentance = self.fixed_sentances["footer"] if self.sentance_gen_type == 'fixed' else self.fake.sentence(nb_words=20)
        lines = break_string_recursively(sentance, 110)
        
        for line in lines:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, 10)
        
        return start_x, start_y
        
    def draw_report(self, report_name: str = 'form_schindler.pdf', image_path = 'form_schindler.jpg'):
        c = Canvas(report_name)
        canvas = c
        self.select_random_font(self.fonts_dir)
        canvas.setFont(self.font, 10)
        canvas.setPageSize(letter)
        canvas.setLineWidth(.5)
        start_x = self.start_x
        start_y = self.start_y
        
        global_keys_key , global_keys_no_key = self.init_global_keys()
        
        global_keys_key = self.populate_keys_key(keys_key=global_keys_key)
        global_keys_no_key = self.populate_keys_nokey(keys_nokey=global_keys_no_key)
        
        #evaluator_addres_object = {"Anschrift des Bewerters":global_keys_no_key["Anschrift des Bewerters"],
        #                           "Tel":global_keys_key["Tel"],
        #                           "Fax": global_keys_key["Fax"]}
        
        evaluator_addres_object = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["evaluator_addres_object"])
        _, new_line = self.draw_Address(canvas = canvas, x = start_x, y = start_y,address=evaluator_addres_object["Anschrift des Bewerters"], font_size=8, line_break=13)
        new_line = next_line(new_line, 10)
        canvas.drawString(start_x, new_line, f'Tel: {evaluator_addres_object["Tel"]} Fax: {evaluator_addres_object["Fax"]}')
        
        new_line = next_line(new_line, random.choice(np.arange(25,35)))
        
        ## Leistungsausweis
        canvas.setFont(self.font, random.choice(np.arange(14,17)))
        canvas.drawString(start_x, new_line, self.fixed_sentances["leistungs_header"])
        new_line = next_line(new_line, 10)
        canvas.setFont(self.font, 10)
        canvas.drawString(start_x, new_line, self.fixed_sentances["leistungs_footer"])
        
        
        ## ADKB
        new_line = next_line(new_line, random.choice(np.arange(25,35)))
        #adkb_object = {key:val for key,val in global_keys_key.items() if key in ("Arbeitsschein-Nr","Datum","Kunden-Nr","Ihre Bestellnummer")}
        adkb_object = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["adkb_object"])
        _, new_line = self.draw_ADKB(canvas = canvas, x=start_x, y = new_line, adkb_object=adkb_object, font_size=8)
        
        ### Owner Address
        new_line = next_line(new_line, random.choice(np.arange(15,25)))
        start_x_temp = start_x
        start_x_temp += random.choice(np.arange(350,400,10))
        _, new_line = self.draw_Address(canvas=canvas, x = start_x_temp, y= new_line, address=global_keys_no_key["Adresse des Besitzers"], font_size=11, line_break = 15)
        
        ## Information anlage
        new_line = next_line(new_line, random.choice(np.arange(15,25)))
        #anst_object = {"Anlagen-Nr":global_keys_key["Anlagen-Nr"], "Standort":global_keys_key["Standort"]}
        anst_object = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["anst_object"])
        key_val_spacing = random.choice(np.arange(120,150,5))
        _,new_line = self.draw_anst(canvas=canvas, x=start_x, y=new_line, font_size=10, anst_object=anst_object, key_val_spacing= key_val_spacing, line_break = random.choice(np.arange(10,15)))
        
        ## Information Auftrag
        new_line = next_line(new_line, random.choice(np.arange(15,25)))

        #afst_object = {"Auftrags-Nr":global_keys_key["Auftrags-Nr"], "Service Techniker":global_keys_key["Service Techniker"]}
        afst_object = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["afst_object"])
        key_val_spacing = random.choice(np.arange(120,150,5))
        _,new_line = self.draw_afst(canvas=canvas, x=start_x, y=new_line, font_size=10, afst_object=afst_object, key_val_spacing= key_val_spacing, line_break = random.choice(np.arange(10,15)))
        
        ## Auftrag sentance
        new_line = next_line(new_line, random.choice(np.arange(10,15)))
        sentance = self.fixed_sentances["auftrag_footer"] if self.sentance_gen_type == 'fixed' else self.generate_ranom_sentance()
        _, new_line = self.draw_auftrag_sentance(canvas=canvas, x=start_x, y=new_line, font_size=9,sentance=sentance )
        
        ## ankunft_vor_ort
        new_line = next_line(new_line, random.choice(np.arange(10,15)))
        #ankunft_object = {"Ankunft vor Ort":global_keys_no_key["Ankunft vor Ort"], "Tätigkeit":global_keys_no_key["Tätigkeit"]}
        ankunft_object = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["ankunft_object"])
        _, new_line = self.draw_ankunft_ort(canvas=canvas, x=start_x, y = new_line, font_size=10, key_val_spacing=key_val_spacing, ankunft_object=ankunft_object, line_break=random.choice(np.arange(10,15)))
        
        
        ## Billing Note
        new_line = next_line(new_line, random.choice(np.arange(15,25)))
        sentance = self.fixed_sentances["billing_note"] if self.sentance_gen_type == 'fixed' else self.generate_ranom_sentance(token_len=10)
        canvas.drawString(start_x, new_line, sentance)
        #print(new_line)
        ## info service rectangle
        
        new_line = next_line(new_line, random.choice(np.arange(20,25)))
        #print(new_line)
        #info_techniker_obj = {"Information Servicetechniker":global_keys_key["Information Servicetechniker"]}
        if random.choice([1,2,3]) == 1:
            info_techniker_obj = self.init_object(global_keys_no_key=global_keys_no_key, global_keys_key = global_keys_key, keys = key_mappings["info_techniker_obj"])
            _, new_line = self.draw_info_technicker_rect(canvas=canvas, x= start_x, y= new_line, font_size= 10,  info_techniker_obj=info_techniker_obj, rect_height=50, rect_width=500, line_break=15)
            new_line = next_line(new_line, random.choice(np.arange(20,25)))
        ## Footer
        
        _, new_line = self.draw_footer(canvas=canvas, x=start_x, y=new_line, line_break=random.choice(np.arange(15,25)), rarity=3, font_size=9)
        
        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(image_path, 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        image = cv2.imread(image_path)[:,:,::-1]
        #global_keys_ext = {'global_keys':global_keys, 'global_keys_config':global_keys_config}
        filtered_dict = self.filter_keys(global_keys_key=global_keys_key, global_keys_nokey= global_keys_no_key)
        #print(filtered_dict)
        global_keys_ext = {'global_keys':filtered_dict, 'global_keys_config':{'font_size': 9}}
        #print(global_keys_ext)
        return global_keys_ext, image
        
if __name__=='__main__':
    start_x = 40
    start_y = [730, 750]
    fonts_dir = "fonts"
    sentances_dir = "sentances_schindler"
    rect_colors = ["lightgrey", "white"]
    sentance_gen_type='fixed'
    keys_to_include = ["Anlagen-Nr", "Standort", "Auftrags-Nr", "Service Techniker"]
     
    
    template_kone = Template_Schindler(start_x=start_x, start_y = start_y, fonts_dir=fonts_dir, rect_colors=rect_colors, keys_to_include = keys_to_include,  sentance_gen_type = sentance_gen_type, sentances_dir=sentances_dir, file_name='form_schindler.pdf')
    _, image = template_kone.draw_report()
    print(image.shape)
    _,_,image = get_ocr_data(image = image)
    #print(type(image))
    cv2.imwrite('schinlder_ocr.png', image)