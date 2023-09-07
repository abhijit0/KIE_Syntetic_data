import os


from Template import Template

import json
#pwd = os.getcwd()
#os.chdir("..")
import sys
sys.path.append('..')

from utility_functions.utilities_kie import *
import pandas as pd

#os.chdir(pwd)
from faker import Faker
locales = ['de_DE']
Faker.seed(0)

class Template_Kone(Template):
    def __init__(self, start_x: int = None, start_y:int = None, token_spacing: int= None
                 , key_val_spacing:int = None, line_break:list = None, file_name:str= None, fonts_dir:str=None, sentances_path:str=None, sentance_gen_type:str=None, fixed_sentance_path:str=None, technician_comments_path:str=None, draw_type:str=None):
        super().__init__()
        self.start_x = start_x
        self.start_y = start_y
        self.token_spacing = token_spacing
        self.key_val_spacing = key_val_spacing
        self.line_break = line_break
        self.file_name = file_name
        self.fake = Faker(locales)
        
        with open(fixed_sentance_path, 'r') as f:
            self.fixed_sentances = json.load(f)
        self.fonts_dir =fonts_dir
        self.sentances = self.convert_df_to_list(sentances_path)
        self.sentance_gen_type = sentance_gen_type
        self.technician_comments = self.convert_df_to_list(technician_comments_path)
        self.draw_type = draw_type
        
    def convert_df_to_list(self,df_path:str, delimiter:str='>'):
        sentances_df = pd.read_csv(df_path, delimiter=delimiter)
        sentance_set = []    
        for _, row in sentances_df.iterrows():
            sentance_set.append(str(row['sentance']).replace('\n', ' '))
        return list(sentance_set)
        
    def select_random_font(self, font_dir:str):
        font = random.choice([f for f in os.listdir(font_dir) if 'bd' not in f[:-4]])
        font = font[:-4]
        self.font =  font
        self.font_bold = f'{self.font}-bold'
        pdfmetrics.registerFont(TTFont(f'{self.font}', f'{font_dir}/{font}.ttf'))
        pdfmetrics.registerFont(TTFont(f'{self.font_bold}', f'{font_dir}/{font}bd.ttf'))
        
    def init_global_keys(self):
        global_keys = {
            'Liegenschaft' : 'KAM_unilimo:Deutschland',
            'Vertragsnummer': '41608279',
            'Straße': 'Grosser Grasbrook 9',
            'Anlagennummer':'FN06026',
            'Objektangabe':'2ER GRUPPE, GRÜN, LI. AZ',
            'Anlagentyp': 'Aufzug',
            'Stadt':'20457 Hamburg',
            'Serviceorder':'687895385',
            'Leistungsdatum': '13/5/2019',
            'Durchgefürt von': 'LEMPA Thorsten'
        }
        global_keys_config = {
            'font_size':9
        }
        return global_keys, global_keys_config
    
    def next_line(self, start_y: int, line_break: int):
        return start_y - line_break
    
    def shuffle_dict(self, dict_):
        dict_items = list(dict_.items())
        random.shuffle(dict_items)
        return dict_items
    
    
    def break_string(self,val, length):
        val_splits = val.split(" ")
        str = ''
        length -= len(val_splits) -1 
        if len(val_splits) < 2:
            return val,'0'
        for i in range(len(val_splits)):
            str+=val_splits[i]
            if len(str) >length:
                #print(f'len(val_splits) {len(val_splits)}')
                #print(f'str {len(str)}')
                #print('---')
                
                line_1 = ' '.join([val_splits[j] for j in range(i)]) if i>=1 else val_splits[0]
                line_2 = ' '.join([val_splits[j] for j in range(i+1, len(val_splits))]) if i+1<len(val_splits) else val_splits[-1]
                return line_1, line_2
        return '0', '0'
        
    
    def get_liegenschaft(self):
        
        company_name = self.fake.company().replace('\n', ' ')
        company_name = ' '.join([company_name.split(' ')[i] for i in range(len(company_name.split(' ')) -1 )])
        return company_name.upper()
    
    def get_vertragsnummer(self, digits:int = 8):
        sequence_from_0 = np.arange(0,9)
        sequence_from_1 = np.arange(1,9)
        vertrags_nummer = str(random.choice(sequence_from_1))
        vertrags_nummer = vertrags_nummer + ''.join([str(random.choice(sequence_from_0)) for _ in range(digits -1)])
        return vertrags_nummer
    
    def get_anlagennummer(self, digits = 7):
        alpha = False
        if random.choice([0,0,0,0,1]) == 1:
            alpha = True
        
        if alpha:
            anlagennumer = ''.join([random.choice(string.ascii_uppercase) for _ in range(random.choice([2,3]))])
            digits -= len(anlagennumer)
            anlagennumer += self.get_vertragsnummer(digits=digits-len(anlagennumer))
        else:
            anlagennumer = self.get_vertragsnummer(digits=digits)
        return anlagennumer
    
    def get_straße(self):
        street = self.fake.street_address().replace('\n', ' ')
        return street
    
    def get_stadt(self):
        stadt = self.fake.city_with_postcode().replace('\n', ' ')
        return stadt
    
    def get_leistungsdatum(self):
        leistungsdatum = self.fake.date(pattern="%d/%m/%Y").replace('\n', ' ')
        return leistungsdatum
    
    def get_durchgefürt_von(self):
        durchgefürt_von = self.fake.last_name().replace('\n', '').upper() +' '+ self.fake.first_name().replace('\n', '')
        return durchgefürt_von
    
    def gen_paragraph(self,type:str, sentance_len:int, club_adjescnet_sentances:bool):
        assert type in ['random', 'predefined']
        if type == 'random':
            paragraph= self.fake.paragraph(nb_sentences=sentance_len).split('.')
            if club_adjescnet_sentances:
                paragraph = [f'{paragraph[i]}. {paragraph[i+1]}' for i in range(len(paragraph)) if i+1 < len(paragraph)]
        else:
            sentance_list = list(self.sentances)
            indices = random.sample(list(np.arange(0,len(sentance_list))), sentance_len)
            paragraph = [sentance_list[i] for i in indices]
        return paragraph    
                       
    def break_string_recursively(self, string:str, length:int):
        lines = []
        string = string.replace("\n", " ")
        line_1, line_2 = self.break_string(string, length)
        lines.append(line_1)
        while(True):
            if len(line_2)<=length :
                lines.append(line_2)
                break
            line_1, line_2 = self.break_string(line_2, length)
            if line_1 == '0' or line_2 == '0':
                break
            lines.append(line_1)
                
        return list(set(lines)) 
    
    def draw_sentances(self,canvas:object=None, x:int=None, y:int=None, font_size:int=None, sentance_len:int=None, type:str=None, club_adjescnet_sentances:bool=None, line_break:list=None, char_len_limit:int=None):
        start_x = x
        start_y = y
        
        line_break = random.choice(line_break)
        paragraph = self.gen_paragraph(type, sentance_len, club_adjescnet_sentances)
        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, "Ausgeführte Arbeiten")
        start_y = self.next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        #paragraph_tokens = [token for sentance in paragraph for token in sentance.split(" ")]
        all_y  = []
        for sentance in paragraph:
            if len(str(sentance)) >char_len_limit:
                lines= self.break_string_recursively(sentance, char_len_limit)
                for line in lines:
                    canvas.drawString(start_x, start_y, line)
                    start_y = self.next_line(start_y, line_break)
                    all_y.append(start_y)
            else: 
                canvas.drawString(start_x, start_y, sentance)
                start_y = self.next_line(start_y, line_break)
                all_y.append(start_y)
                    
        return start_x, min(all_y)
                
        
        
    def draw_key_vals_four_column(self, canvas:object=None, x:int=None, y:int=None, global_keys:dict=None, global_keys_config:dict=None, line_break:list=None):
        start_x = x 
         
        global_keys_shuffled = self.shuffle_dict(global_keys)
        canvas.setFont(self.font_bold, global_keys_config['font_size'] + 9)
        canvas.drawString(start_x, y, 'KONE Wartungsnachweis')
        y = self.next_line(y, 40)
        start_y = y
        line_break = random.choice(line_break)
        all_y = []
        for i, (key, val) in enumerate(global_keys_shuffled):
            canvas.setFont(self.font_bold, global_keys_config['font_size'])
            canvas.drawString(start_x, start_y, key)
            #start_x += self.key_val_spacing
            canvas.setFont(self.font, global_keys_config['font_size'])
            if len(val)>23:
                val_line1, val_line2 = self.break_string(val, 22)
                canvas.drawString(start_x + self.key_val_spacing, start_y, val_line1)
                if val_line2 != '0':
                    start_y= self.next_line(start_y, 12)
                    all_y.append(start_y)
                    canvas.drawString(start_x + self.key_val_spacing, start_y, val_line2)
            else:    
                canvas.drawString(start_x + self.key_val_spacing, start_y, val)
            start_y = self.next_line(start_y, line_break)
            all_y.append(start_y)
            
            if i == len(global_keys_shuffled) //2 - 1:
                start_y = y
                start_x += int(self.key_val_spacing * 2.1)
        
        return start_x, min(all_y)
    
    def draw_key_vals_two_column(self, canvas:object=None, x:int=None, y:int=None, global_keys:dict=None, global_keys_config:dict=None, line_break:list=None):
        start_x = x 
        start_y = y 
        
        global_keys_shuffled = self.shuffle_dict(global_keys)
        
        canvas.setLineWidth(.7)            
        line_break = random.choice(line_break)
        all_y = []
        for i, (key, val) in enumerate(global_keys_shuffled):
            canvas.setFont(self.font_bold, global_keys_config['font_size'])
            canvas.drawString(start_x, start_y, key)
            canvas.setFont(self.font, global_keys_config['font_size'])
            canvas.drawString(start_x+3*self.key_val_spacing, start_y, val)
            start_y = self.next_line(start_y, 12)
            all_y.append(start_y)
            if i in (1,8):
                canvas.line(start_x, start_y, 560, start_y)
                start_y = self.next_line(start_y, 12)
                all_y.append(start_y)
        return start_x, min(all_y)
    
    def populate_global_keys(self, global_keys:dict):
        global_keys['Liegenschaft'] = self.get_liegenschaft()
        global_keys['Anlagennummer'] = self.get_anlagennummer()
        global_keys['Vertragsnummer'] = self.get_vertragsnummer()
        global_keys['Leistungsdatum'] = self.get_leistungsdatum()
        global_keys['Straße'] = self.get_straße()
        global_keys['Stadt'] = self.get_stadt()
        global_keys['Serviceorder'] = self.get_vertragsnummer(digits=9)
        global_keys['Durchgefürt von'] = self.get_durchgefürt_von()
        return global_keys
    
    def draw_techniker_comments_four_column(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, line_break:list=None):
        start_x = x
        start_y = y
        
        line_break = random.choice(line_break)
        canvas.setFont(self.font_bold, font_size)
        
        canvas.drawString(start_x, start_y, self.fixed_sentances["techniker_comments"])
        start_y = self.next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        
        if self.sentance_gen_type != 'random':
            sentance = random.choice(self.technician_comments)
        else:
            sentance = self.fake.sentance(nb_words=random.choice(np.arange(10,15)))
        
        char_len_limit = 60
        if len(sentance) < char_len_limit:
            canvas.drawString(start_x, start_y, sentance)
        else:
            line_1, line_2 = self.break_string(sentance, char_len_limit)
            if line_1 is not None:
                canvas.drawString(start_x, start_y, line_1)
                start_y = self.next_line(start_y, 10)
            if line_2 is not None:
                canvas.drawString(start_x, start_y, line_2)
            else:
                canvas.drawString(start_x, start_y, sentance)
        return start_x, start_y
    
    def draw_techniker_comments_two_column(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, line_break:list=None):
        start_x = x
        start_y = y
        
        line_break = random.choice(line_break)
        canvas.setFont(self.font_bold, font_size)
        techniker_comments_line_1 = ' '.join(i for i in self.fixed_sentances["techniker_comments"].split(" ")[:2])
        techniker_comments_line_2 = ' '.join(i for i in self.fixed_sentances["techniker_comments"].split(" ")[2:])
        canvas.drawString(start_x, start_y, techniker_comments_line_1)
        start_y = self.next_line(start_y, line_break)
        canvas.drawString(start_x, start_y, techniker_comments_line_2)
        start_y = y
        canvas.setFont(self.font, font_size -1 )
        
        start_x += 200
        technician_comment_split = random.choice(self.technician_comments).split(" ")
        
        canvas.drawString(start_x, start_y, ' '.join([technician_comment_split[i] for i in range(0, len(technician_comment_split)//2)]))
        start_y = self.next_line(start_y, line_break)
        canvas.drawString(start_x, start_y, ' '.join([technician_comment_split[i] for i in range(len(technician_comment_split)//2, len(technician_comment_split))]))
        return start_x, start_y
    
    
    def generate_telephone_number(self):
        area_code = self.fake.random_int(min=2, max=99)
        subscriber_number = self.fake.random_number(digits=5)
        suffix_number = random.choice([self.fake.random_number(digits=3), self.fake.random_number(digits=1)])
        telephone_number = f"+49.{area_code}.{subscriber_number}-{suffix_number}"
        return telephone_number.replace('\n', '')
    
    def draw_footer(self, canvas:object=None, x:int=None, y:int=None, paragraph:str=None, font_size:int=None, char_len_limit:int=None):
        #sentances = paragraph.split(".")
        start_x = x
        start_y = y
        canvas.setFont(self.font, font_size)
        paragraph = paragraph.replace('\n', ' ')
        #print(len(paragraph))
        #lines = self.break_string_recursively(paragraph, 100)
        lines = self.break_string_recursively(paragraph, char_len_limit)
        for sentance in lines:
            canvas.drawString(start_x, start_y, sentance)
            start_y = self.next_line(start_y, 8)
        
        start_y = self.next_line(start_y, 25)
        if self.sentance_gen_type != 'random':
            company_name = self.fake.company().replace('\n', ' ')
            company_email = self.fake.email().replace('\n', ' ')
            company_address = company_name +' '+self.fake.address().replace("\n", " ")
            telephone_fax = 'Telefonnr.:' + str(self.generate_telephone_number())+' / Faxnr.:' + str(self.generate_telephone_number())
                
        else:
            company_name = self.fixed_sentances['company_name_footer']
            company_email = self.fixed_sentances['company_email']
            company_address = self.fixed_sentances['company_address']
            telephone_fax = self.fixed_sentances['company_ph_fax']
            
        canvas.drawString(start_x, start_y, company_name)
        start_y = self.next_line(start_y, 8)
        canvas.drawString(start_x, start_y, company_email)
        start_y = self.next_line(start_y, 25)
        canvas.drawString(start_x, start_y, company_address)
        start_y = self.next_line(start_y, 8)
        canvas.drawString(start_x, start_y, telephone_fax)
        
        
        
        return start_x, start_y
    
    
    def draw_report_two_column(self,canvas:object=None, global_keys:dict=None, global_keys_config:dict=None, sentance_len:int=None, club_adjescnet_sentances:bool=None, char_len_limit:int=None):
        global_keys_config['font_size']+=2
        ## Draw key vals
        start_x = self.start_x
        start_y = self.start_y
        _, new_line = self.draw_key_vals_two_column(canvas=canvas, x = start_x, y = start_y, global_keys=global_keys, global_keys_config = global_keys_config, line_break = self.line_break)
        line_break = [i-4 for i in self.line_break]
        
        ## Draw sentances
        _, new_line = self.draw_sentances(canvas=canvas, x=start_x, y = new_line, font_size=global_keys_config['font_size']-1, 
                                          sentance_len=sentance_len, type=self.sentance_gen_type, 
                                          club_adjescnet_sentances=club_adjescnet_sentances, line_break=line_break, char_len_limit=char_len_limit)
        
        ## Draw technician comments
        new_line_gap = random.choice(np.arange(10,15))
        new_line = self.next_line(new_line, new_line_gap)
        _, new_line = self.draw_techniker_comments_two_column(canvas=canvas, x=start_x, y = new_line, font_size= global_keys_config['font_size'], 
                                                   line_break = line_break)
        new_line= self.next_line(new_line, new_line_gap)
        canvas.setLineWidth(.7)
        canvas.line(start_x, new_line, 580, new_line)
        
        ## Draw kunden name
        new_line = self.next_line(new_line, new_line_gap * 2)
        canvas.setFont(self.font_bold, global_keys_config["font_size"]+1)
        canvas.drawString(start_x, new_line, self.fixed_sentances["customer_signature"])
        
        new_line = self.next_line(new_line, int(new_line_gap * 2.5))
        canvas.setFont(self.font_bold, global_keys_config["font_size"]+1)
        canvas.drawString(start_x, new_line, self.fixed_sentances["customer_name"])
        
        new_line = self.next_line(new_line, new_line_gap//2)
        
        new_line = self.next_line(new_line, new_line_gap//2)
        
        self.draw_footer(canvas= canvas, x=start_x, y = new_line, paragraph=self.fixed_sentances["footer"], 
                         font_size=global_keys_config['font_size']-2, char_len_limit=130)
        
        ## Draw Footer
        
    
    def draw_report_four_column(self,canvas:object=None, global_keys:dict=None, global_keys_config:dict=None, sentance_len:int=None, club_adjescnet_sentances:bool=None, char_len_limit:int=None):
        ## four column draw format
        start_x = self.start_x
        start_y = self.start_y
        canvas.line(self.start_x, self.start_y, 570, self.start_y)
        start_y = self.next_line(start_y, 30)
        _, new_line = self.draw_key_vals_four_column(canvas= canvas, x = start_x, y= start_y, global_keys=global_keys, 
                                                     global_keys_config = global_keys_config, line_break=self.line_break)
        canvas.setLineWidth(.5)
        canvas.line(start_x, new_line, 450, new_line)
        
        new_line_gap = random.choice(np.arange(25,35))
        new_line = self.next_line(new_line, new_line_gap)
        
        ## generating sentances 
        line_break = [i-4 for i in self.line_break]
        _, new_line = self.draw_sentances(canvas = canvas, x = start_x, y = new_line, font_size= global_keys_config['font_size'], sentance_len = sentance_len, 
                                          type = self.sentance_gen_type, club_adjescnet_sentances=club_adjescnet_sentances, 
                                          line_break=line_break, char_len_limit= char_len_limit)
        
        
        ## generating fixed sentances (arbeit)
        new_line = self.next_line(new_line, new_line_gap)
        canvas.drawString(start_x, new_line, self.fixed_sentances['arbeiten'])
        
        ## Generating technkiker commnets
        new_line = self.next_line(new_line, new_line_gap)
        _, new_line = self.draw_techniker_comments_four_column(canvas=canvas, x=start_x, y = new_line, font_size= global_keys_config['font_size'], 
                                                   line_break = line_break)
        
        ## Kunden Name
        new_line = self.next_line(new_line, new_line_gap * 2)
        canvas.setFont(self.font_bold, global_keys_config["font_size"]+1)
        canvas.drawString(start_x, new_line, self.fixed_sentances["customer_signature"])
        
        new_line = self.next_line(new_line, int(new_line_gap * 2.5))
        canvas.setFont(self.font_bold, global_keys_config["font_size"]+1)
        canvas.drawString(start_x, new_line, self.fixed_sentances["customer_name"])
        
        
        new_line = self.next_line(new_line, new_line_gap//2)
        canvas.setLineWidth(.5)
        canvas.line(start_x, new_line, 580, new_line)
        
        ## Footer
        new_line = self.next_line(new_line, new_line_gap//2)
        
        self.draw_footer(canvas=canvas, x=start_x, y=new_line, paragraph=self.fixed_sentances["footer"], font_size=global_keys_config['font_size']-2, char_len_limit=170)
        
    def draw_report(self, report_name: str = 'form_kone.pdf', image_path = 'form_kone.jpg'):
        c = Canvas(report_name)
        canvas = c
        self.select_random_font(self.fonts_dir)
        canvas.setFont(self.font, 10)
        canvas.setPageSize(letter)
        canvas.setLineWidth(.5)
        
        
        canvas.setFillColor(HexColor(0x000000))
    
        
        global_keys, global_keys_config = self.init_global_keys()
        global_keys = self.populate_global_keys(global_keys)
        
        ## Deciding the sentance gen type
        club_adjescnet_sentances = True if self.sentance_gen_type == 'random' else False
        
        ## two column draw format
        if self.draw_type == '2c':
            char_len_limit = 100
            self.draw_report_two_column(canvas=canvas, global_keys=global_keys, global_keys_config=global_keys_config, sentance_len=10 ,club_adjescnet_sentances = club_adjescnet_sentances, char_len_limit= char_len_limit)
        else:
            char_len_limit = 80
            self.draw_report_four_column(canvas=canvas, global_keys=global_keys, global_keys_config = global_keys_config, sentance_len= 10, club_adjescnet_sentances = club_adjescnet_sentances, char_len_limit = char_len_limit)
        
        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(image_path, 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        image = cv2.imread(image_path)[:,:,::-1]
        global_keys_ext = {'global_keys':global_keys, 'global_keys_config':global_keys_config}
        return global_keys_ext, image

    '''def draw_report(self, report_name: str = 'form_kone.pdf', image_path = 'form_kone.jpg'):
        c = Canvas(report_name)
        canvas = c
        self.select_random_font('fonts/')
        canvas.setFont(self.font, 10)
        canvas.setPageSize(letter)
        canvas.setLineWidth(.5)
        
        
        canvas.setFillColor(HexColor(0x000000))
    
        
        global_keys, global_keys_config = self.init_global_keys()
        global_keys = self.populate_global_keys(global_keys)
        
        ## Deciding the sentance gen type
        club_adjescnet_sentances = True if self.sentance_gen_type == 'random' else False
        
        ## two column draw format
        if self.draw_type == '2c':
            global_keys_config['font_size']+=2
            _, new_line = self.draw_key_vals_two_column(canvas, self.start_x, self.start_y, global_keys,global_keys_config, self.line_break)
            
        else:
            ## four column draw format
            canvas.line(self.start_x, self.start_y, 570, self.start_y)
            self.start_y = self.next_line(self.start_y, 40)
            _, new_line = self.draw_key_vals_four_column(canvas, self.start_x, self.start_y, global_keys,global_keys_config, self.line_break)
            canvas.setLineWidth(.3)
            canvas.line(start_x, new_line, 450, new_line)
        
        new_line_gap = random.choice(np.arange(25,35))
        new_line = self.next_line(new_line, new_line_gap)
        
        ## generating sentances 
        line_break = [i-4 for i in self.line_break]
        _, new_line = self.draw_sentances(canvas, start_x, new_line, 10, 10, self.sentance_gen_type, global_keys_config, club_adjescnet_sentances, line_break)
        
        
        ## generating fixed sentances (arbeit)
        new_line = self.next_line(new_line, new_line_gap)
        canvas.drawString(start_x, new_line, self.fixed_sentances['arbeiten'])
        
        ## Generating technkiker commnets
        new_line = self.next_line(new_line, new_line_gap)
        _, new_line = self.draw_techniker_comments(canvas, start_x, new_line, global_keys_config, line_break)
        
        ## Kunden Name
        new_line = self.next_line(new_line, new_line_gap * 2)
        canvas.setFont(self.font_bold, global_keys_config["font_size"]+1)
        canvas.drawString(start_x, new_line, self.fixed_sentances["customer_name"])
        new_line = self.next_line(new_line, new_line_gap//2)
        canvas.setLineWidth(.3)
        canvas.line(start_x, new_line, 580, new_line)
        
        ## Footer
        new_line = self.next_line(new_line, new_line_gap//2)
        
        self.draw_footer(canvas, start_x, new_line, self.fixed_sentances["footer"], global_keys_config['font_size']-2)
        
        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(image_path, 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        image = cv2.imread(image_path)[:,:,::-1]
        return global_keys, image'''
        

if __name__=='__main__':
    start_x = 20
    start_y = random.choice(np.arange(730, 750, 5))
    line_break = np.arange(15,18)
    key_val_spacing = 115
    draw_type='4c'
    sentance_path = 'sentances_kone/sentances.csv'
    fixed_sentance_path = 'sentances_kone/fixed_sentances.json'
    technician_comments_path = 'sentances_kone/technician_comments_kone.csv'
    sentance_gen_type='predefined'
  
    for _ in range(40):
        template_kone = Template_Kone(start_x=start_x, start_y=start_y, line_break=line_break, key_val_spacing=key_val_spacing, sentances_path = sentance_path, sentance_gen_type = sentance_gen_type, fixed_sentance_path = fixed_sentance_path, technician_comments_path = technician_comments_path, draw_type=draw_type)
        template_kone.draw_report()
    