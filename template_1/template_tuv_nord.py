
import os
from typing import Any


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


class Template_TUV_Nord(Template):

    header_str_fixed = "Zugelassene Überwachungsstelle de"
    def __init__(self, start_x:int, start_y:int, shuffle_dict:bool=None, sentance_gen_type:str=None, sentances_dir:str=None,keys_to_include:list=None, 
                 file_name:str= None, fonts_dir:str=None,random_numbers:bool=None):
        super().__init__()
        self.start_x =start_x
        self.start_y = start_y
        self.shuffle_dict = shuffle_dict
        self.sentance_gen_type = sentance_gen_type
        self.sentances_dir= sentances_dir
        with open(self.sentances_dir, 'r') as f:
            self.fixed_sentances = json.load(f)
        self.keys_to_include = keys_to_include
        self.file_name = file_name
        self.fonts_dir = fonts_dir
        self.fake = Faker(locales)
        self.random_numbers= random_numbers
        self.include_pruf_interval = random.choice([*[0 for _ in range(10)], *[1 for _ in range(70)], *[2 for _ in range(20)]])

    def init_key_mappings(self):
        key_mappings = {
            'prufbescheinigung_object' : ['Equipment-Nummer', 'Auftrags-Nummer', 'Kunden-Nummer', 'Akten-Nummer'],
            'prufbescheinigung_object_footer' : ['Für Sie vor Ort', 'Telefon', 'Fax', 'e-mail'],
            'personaufzug_object' : ['Leistungsort', 'Leistungsempfänger'],
            'techniche_daten_2c' : ['Baujahr', 'Tragfähigkeit', random.choice(['Betriebsteilkurzbegriff', 'Standort b.kunden']), 'Hersteller', 
                            random.choice(['Fabrik-Nr', 'Fabrik-/Herstell-Nr']), random.choice(['Ausführende Fa.', 'Ausführende Firma'])],
            'techniche_daten_2c_all' : ['Baujahr', 'Tragfähigkeit', 'Betriebsteilkurzbegriff', 'Standort b.kunden', 'Hersteller', 
                            'Fabrik-Nr', 'Fabrik-/Herstell-Nr', 'Ausführende Fa.', 'Ausführende Firma'],
            'techniche_daten_3c' : ['Fabrik-/Herstell-Nr', 'Inventar-/Werk-Nr', 'Hersteller', 'Baujahr', 
                                'Betriebsteil', 'Ausführende Fa.', 'Tragfähigkeit', "Kostenst.Betreiber"],
            'prüfgrundlage' : [random.choice(['Errichtungsvorschrift', 'Prüfgrundlage'])],
        'festgestellte_mangel' : ['Festgestellte Mängel'],
        "prufergebnis_interval" : ["Prüfintervall", "Nächste Hauptprüfung", "Nächste Zwischenprüfung"],
        "prufort_object":["Prüfort", "Prüfdatum", "Sachverständige(r)"]
            }
        return key_mappings


    def init_global_keys(self):
        global_keys_no_key = {
            "header" : "Zugelassene Überwachungsstelle der TÜV NORD Systems GmbH & Co. KG Postfach 54 02 20 – 22502 Hamburg",
            "evaulator_address" :"Union Investment Real Estate GmbH c/o Strabag Property and Facility Services GmbH Valentinskamp 70 20355 Hamburg",

        }

        global_keys_key = {
            "Equipment-Nummer" : "0010278591",
            "Auftrags-Nummer" : "8111136784-0110",
            "Kunden-Nummer" : "0055228514",
            "Akten-Nummer" : "88206-035421",
            "Für Sie vor Ort" : "TÜV NORD SYSTEMS GMBH & CO.KG GESCHÄFTSSTELLE HAMBURG REGION HAMBURG SÜD•GROßE BAHNSTR. 31•22525•HAMBURG",
            "Telefon" : "+49 40 8557 - 2053",
            "Fax" : "+49 40 8557 - 2655",
            "e-mail" : "RegionHamburgSued@tuev-nord.de",
            "Leistungsort" : "Emporio Tower Dammtorwall 15 20355 Hamburg",
            "Leistungsempfänger" : "Union Investment Real Estate GmbH c/o Strabag Property and Facility Services GmbH Valentinskamp 70 20355 Hamburg",
            "Fabrik-/Herstell-Nr" : "1118370",
            "Baujahr":"2010",
            "Tragfähigkeit": "1350 kg",
            "Betriebsteil":"Aufzug D",
            "Fabrik-Nr": "88938",
            "Standort b.kunden": "N.A",
            "Inventar-/Werk-Nr":"N.A",
            "Hersteller": "Schindler Aufzüge",
            "Ausführende Fa." : "Schindler Aufzüge",
            "Prüfgrundlage":"BetrSichV in Verbindung mit TRBS 1201 Teil 4",
            "Festgestellte Mängel":"Keine",
            "Erläuterungen/Hinweise" : "Keine",
            "Prüfergebnis": "ohne Mängel",
            "Prüfintervall" : "24 Monate",
            "Nächste Hauptprüfung": "01.2020",
            "Nächste Zwischenprüfung" : "01,2021",
            "Prüfort" : "Hamburg",
            "Prüfdatum" : "08.05.2014",
            "Sachverständige(r)" : "gez. Andreas Krampf",
            "Betriebsteilkurzbegriff":"N.A",
            "Ausführende Firma": "Matthiessen, K.+M.",
            "Kostenst.Betreiber" : "N.A",
            "Errichtungsvorschrift" : "TRA 200",
            "Prüfgrundlage" : "BetrSichV in Verbindung mit TRBS 1201 Teil 4",
            "Festgestellte Mängel" : "Keine",
            "Erläuterungen/Hinweise" : "Keine"
            

        }

        global_keys_no_key = {
            "Anschrift des Bewerters" :"TÜV NORD Systems GmbH & Co. KG Postfach 54 02 20 – 22502 Hamburg",
            "Adresse des Besitzers" : "Union Investment Real Estate GmbH Valentinskamp 70 20355 Hamburg",
        }

        return global_keys_key, global_keys_no_key
    
    def populate_global_keys(self, global_keys_key:dict=None, global_keys_no_key:dict=None):
        global_keys_key['Equipment-Nummer'] = self.generate_equipment_nummer()
        global_keys_key['Auftrags-Nummer'] = self.generate_auftragsnummer()
        global_keys_key['Kunden-Nummer'] = self.generate_kunden_nummer()
        global_keys_key['Akten-Nummer'] = self.generate_akten_nummer()
        global_keys_key['Telefon'] = self.generate_telephone()
        global_keys_key['Fax'] = self.generate_telephone()
        global_keys_key['e-mail'] = self.fake.email().replace('\n', ' ')
        leistungs_ort= self.generate_leistungs_ort()
        global_keys_key['Leistungsort'] = leistungs_ort
        
        global_keys_key['Baujahr'] = self.generate_baujarh()
        global_keys_key['Tragfähigkeit'] = self.generate_tregfahigkeit()
        fabrik_nr, digit_ch = self.generate_fabrik_nr()
        if digit_ch == 9:
            global_keys_key['Fabrik-Nr'] = f'{fabrik_nr}/{global_keys_key["Baujahr"]}'
            global_keys_key['Fabrik-/Herstell-Nr'] = f'{fabrik_nr}/{global_keys_key["Baujahr"]}'
        else:
            global_keys_key['Fabrik-Nr'] = fabrik_nr
            global_keys_key['Fabrik-/Herstell-Nr'] = fabrik_nr

        global_keys_key["Errichtungsvorschrift"] = random.choice(self.fixed_sentances["errichtungsvorschrift"])
        global_keys_key["Prüfgrundlage"] = random.choice(self.fixed_sentances["prufgrundlage"])
        global_keys_key["Betriebsteil"] = self.generate_betriebsteil()
        global_keys_key['Hersteller'] = self.fake.company().replace('\n', ' ')
        global_keys_key['Ausführende Fa.'] = global_keys_key['Hersteller']
        global_keys_key['Ausführende Firma'] = global_keys_key['Ausführende Fa.']
        global_keys_key["Festgestellte Mängel"] = self.generate_festgestellte_mangel()
        global_keys_key["Prüfergebnis"] = "ohne Mängel" if global_keys_key["Festgestellte Mängel"][0].lower() == 'keine' else "geringe Mängel"
        
        pruf_interval, nachte_hp, nachte_zp = self.generate_pruf_interval()
        global_keys_key['Prüfintervall'] = pruf_interval
        global_keys_key['Nachste Hauptprüfung'] = nachte_hp
        global_keys_key['Nächste Zwischenprüfung'] = nachte_zp
        global_keys_key['Prüfort'] = self.fake.city().replace('\n', ' ')
        day = random.choice(np.arange(1,31))
        day = '0'+str(day) if day<10 else day 
        global_keys_key['Prüfdatum'] = f'{day}.{nachte_hp}'
        global_keys_key['Sachverständige(r)'] = self.generate_fake_name()

        global_keys_no_key['Anschrift des Bewerters'] = self.generate_company_adress()
        global_keys_no_key['Adresse des Besitzers'] = self.generate_company_adress()
        global_keys_key['Für Sie vor Ort'] = global_keys_no_key['Anschrift des Bewerters']
        global_keys_key['Leistungsempfänger'] = global_keys_no_key['Adresse des Besitzers']

        return global_keys_key, global_keys_no_key

        
    def generate_fake_name(self):
        prefix = self.fake.prefix().replace('\n', ' ')
        first_name = self.fake.first_name().replace('\n', ' ')
        last_name = self.fake.last_name().replace('\n', ' ')

        return f'{prefix} {first_name} {last_name}'

    def generate_company_adress(self): # stands for Anschrift des Bewerters
        company_name = self.fake.company().replace('\n', ' ')
        comapny_address = self.fake.address().replace('\n',' ')
        company_city = self.fake.city().replace('\n', ' ')
        company_postcode = self.fake.postcode().replace('\n', ' ')
         
        return f'{company_name} {comapny_address} {company_postcode} {company_city}'
    
    def generate_leistungs_ort(self):
        ort = self.fake.street_address().replace('\n', ' ')
        postcode = self.fake.postcode().replace('\n', ' ')
        city = self.fake.city().replace('\n', ' ')

        leistungs_ort = f'{ort} {postcode} {city}'
     
        return leistungs_ort

    def generate_equipment_nummer(self):
        if self.random_numbers:
            return ''.join([random.choice(string.digits) for _ in range(random.choice([8,9,10,11]))])
        part1 = random.choice(['0010', '100', '10', '001'])
        return part1+str(''.join([random.choice(string.digits) for _ in range(random.choice([5,6,7,8]))]))
    
    def generate_auftragsnummer(self):
        if self.random_numbers:
            part_1 = ''.join([random.choice(string.digits) for _ in range(10)])
            part_2 = ''.join(['0' for i in range(random.choice([4,5,6]))])
            return f'{part_1}-{part_2}'
        
        part1 = ''.join([random.choice(string.digits) for _ in range(8)])
        part1 = '81' + part1
        part2 = ''.join(['0' for i in range(random.choice([1,2,3]))]) + '110'

        return f'{part1}-{part2}'
    
    def generate_kunden_nummer(self):
        if self.random_numbers:
            return ''.join([random.choice(string.digits) for _ in range(random.choice([8,10]))])
        part1 = random.choice(['55', '0055'])
        part2 = ''.join([random.choice(string.digits) for _ in range(6)])
        return part1+part2
    
    def generate_akten_nummer(self):
        add_prefix = random.choice([i<4 for i in range(100)])
        if self.random_numbers:
            part_1= generate_random_digit_string(5)
            part_2= generate_random_digit_string(6)
  
        else:
            digit_prefix_part1 = ['88', '86', '50']
            digit_prefix_part2 = ['206', '208']
            digit_prefix_part3 = ['0', '9']
            
            part_1 = random.choice(digit_prefix_part1)+random.choice(digit_prefix_part2)
            part_2 = random.choice(digit_prefix_part3) + generate_random_digit_string(5)
        
        if add_prefix:
            akten_nummer = part_2[1:] +' '+ part_1 +'-'+ part_2
        else:
            akten_nummer = part_1 + '-' + part_2

        return akten_nummer
    
    def generate_telephone(self):
        area_code = self.fake.random_int(min=2, max=99)
        subscriber_number = self.fake.random_number(digits=5)
        suffix_number = random.choice([self.fake.random_number(digits=3), self.fake.random_number(digits=1)])
        telephone_number = f"+49 {area_code} {subscriber_number}-{suffix_number}"
        return telephone_number.replace('\n', '')
    

    def generate_baujarh(self):
        baujahren = np.arange(1990, 2023)
        baujahr = random.choice(baujahren)
        return str(baujahr)
    
    def generate_tregfahigkeit(self):
        weight = np.arange(500,4000,50)
        return f'{random.choice(weight)} kg'
    
    def generate_fabrik_nr(self):
        digits = [5,7,9]
        prefix_5 = '88'
        prefix_7 = random.choice([random.choice([str(i) for i in np.arange(100, 112)]), '529'])
        prefix_9 = random.choice(['62'+str(i) for i in np.arange(20,30)])
        
        prefix_map = {5:prefix_5, 7:prefix_7, 9:prefix_9}

        digit_choice = random.choice(digits)

        if self.random_numbers:
            if digit_choice == 9:
                fabrik_nr = f'{generate_random_digit_string(4)}/0{generate_random_digit_string(1)}'

            else:
                fabrik_nr = generate_random_digit_string(digit_choice) 
        else:
            if digit_choice == 9:
                fabrik_nr = f'{prefix_9}/0{generate_random_digit_string(1)}'
            elif digit_choice == 5:
                fabrik_nr = f'{prefix_5}{generate_random_digit_string(3)}'
            else:
                fabrik_nr = f'{prefix_7}{generate_random_digit_string(4)}'
        
        return fabrik_nr, digit_choice
    
    def generate_betriebsteil(self):
        empty_occurrences = [i<20 for i in range(100)]
        type1 = f'Aufzug {random.choice(string.ascii_uppercase)}'
        type2 = 'Pü >150 bar'
        type3 = f'{random.choice(["rechter", "linker"])} Aufzug'

        if random.choice(empty_occurrences):
            betreibsteil = 'N.A'
        else:
            betreibsteil = random.choice([type1, type2, type3])
        
        return betreibsteil
    
    def generate_pruf_interval_object(self, global_keys_key :dict=None, key_mapping:list=None):
        pruf_interval_object = extract_object(global_object=global_keys_key, filter_keys= key_mapping)
        pruf_interval_object = [(key,val) for key,val in pruf_interval_object.items()]
        if self.include_pruf_interval == 0 :
            return []
        elif self.include_pruf_interval == 1:
            return [pruf_interval_object[0]]
        else:
            return pruf_interval_object
        
    def generate_pruf_interval(self):
        month_interval = random.choice(np.arange(12,48, 12))
        months = np.arange(1,13)
        month_pruf = random.choice(months)
        mm = '0'+str(month_pruf) if month_pruf<10 else str(month_pruf)
        yy = random.choice(np.arange(1990, 2023))

        pruf_interval = f'{month_interval} Monate'

        naschte_hp = f'{mm}.{yy}'

        mid_interval = month_pruf / (12 * 2)
        if mid_interval < 1.0:
            mm_zp = month_pruf + 6
            if mm_zp >12:
                mm_zp = mm_zp % 12 
                yy_zp = yy + 1
            else:
                yy_zp = yy
        else:
            mm_zp = month_pruf
            yy_zp = yy + 1
        mm_zp = '0'+str(mm_zp) if month_pruf<10 else str(mm_zp)
        naschte_zp = f'{mm_zp}.{yy_zp}'

        return pruf_interval, naschte_hp, naschte_zp

        
    def generate_festgestellte_mangel(self):
        is_mangel = random.choice([i<50 for i in range(100)])

        if is_mangel:
            return ['Keine']
        else:
            mange_count = random.choice(np.arange(1,4))
            mangel_list = random.choices(self.fixed_sentances["festgestellte_mangel"], k= mange_count)
            return mangel_list



    def draw_header(self, canvas:object=None, x:int=None, y:int=None, header_str:dict=None, font_size:int=None, line_break:int=8):
        start_x = x
        start_y = y

        canvas.setFont(self.font, font_size)
        header_string = f'{Template_TUV_Nord.header_str_fixed} {header_str}'

        header_lines = break_string_recursively(header_string, 60)
        for line in header_lines:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, line_break)

        return start_x, start_y
    

    def draw_owner_address(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, owner_address:str=None, line_break:int=None):
        start_x=x
        start_y=y

        canvas.setFont(self.font, font_size)
        lines = break_string_recursively(owner_address, 25)
        for line in lines:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, line_break)
        
        return start_x, start_y

    def draw_Prüfbescheinigung(self, canvas:object=None, x:int=None, y:int=None, font_size:int=None, line_break:int=None, prufbescheinigung_object:dict=None, 
                               rect_width:int=None, rect_height:int=None):
        start_x = x
        start_y = y
        
        #prufbescheinigung_object = {key:val for key,val in global_keys_key.items() if key in prufbescheinigung_object_keys}

        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(prufbescheinigung_object)
        else:
            shuffled_dict = [(key, val) for key,val in prufbescheinigung_object.items()]

        canvas.setFont(self.font, font_size+2)
        canvas.drawString(start_x, start_y, 'Prüfbescheinigung')
        canvas.setFillColor('white')
        
        start_y = next_line(start_y, 25)
        canvas.rect(start_x, start_y, rect_width, -1 * rect_height,stroke=1, fill=0)
        
        start_y = next_line(start_y, 15)
        canvas.setFillColor('black')
        canvas.setFont(self.font, font_size)
        
        temp = start_x + 5
        start_x_temp = temp
        canvas.drawString(start_x_temp, start_y, 'Bei Rückfragen bitte immer angeben')
        start_y = next_line(start_y, line_break)

        
        for i, (key,val) in enumerate(shuffled_dict):
            canvas.drawString(start_x_temp, start_y, key)
            start_x_temp+=100
            if i == len(shuffled_dict) -1 :
                start_x_temp+=random.choice(np.arange(40,60))
            canvas.drawString(start_x_temp, start_y, val)
            start_y = next_line(start_y, line_break)
            start_x_temp = temp
        
        return start_x_temp, start_y


    def draw_Prüfbescheinigung_footer(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, footer_object:dict=None, font_size:int=None):
        start_x = x
        start_y = y

        canvas.setFont(self.font, font_size)
        if self.shuffle_dict:
            shuffled_dict= shuffle_dict(footer_object)
        else:
            shuffled_dict = [(key,val) for key,val in footer_object.items()]
        
        for i, (key, val) in enumerate(shuffled_dict):
            key_val = f'{key}: {val}'
            lines = break_string_recursively(key_val, 75)
            for line in lines:
                canvas.drawString(start_x , start_y, line)
                start_y = next_line(start_y, line_break)
            
        return start_x, start_y
    
    
    

    def draw_personeaufzug(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, font_size:int=None, personeafuzug_object:dict=None):
        start_x = x
        start_y = y

        canvas.setFont(self.font_bold, font_size+1)
        canvas.drawString(start_x, start_y, 'Personenaufzug')

        start_y = next_line(start_y, 20)
        start_y_temp = start_y #used for second key
        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(personeafuzug_object)
        else:
            shuffled_dict = [(key, val) for key,val in personeafuzug_object.items()]

        canvas.setFont(self.font, font_size)
        
        canvas.drawString(start_x, start_y, shuffled_dict[0][0])
        start_y = next_line(start_y, 1)
        canvas.line(start_x, start_y, start_x+40, start_y)
        start_y = next_line(start_y, line_break + 1)
        lines_1 = break_string_recursively(shuffled_dict[0][1], 20)
        for line in lines_1:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, line_break)

        start_x += random.choice(np.arange(200,240))
        start_y = start_y_temp
        canvas.drawString(start_x, start_y, shuffled_dict[1][0])
        start_y = next_line(start_y, 1)
        canvas.line(start_x, start_y, start_x+60, start_y)
        start_y = next_line(start_y, line_break + 1)
        lines_1 = break_string_recursively(shuffled_dict[1][1], 30)
        for line in lines_1:
            canvas.drawString(start_x, start_y, line)
            start_y = next_line(start_y, line_break)

    

        return start_x, start_y
    
    def draw_techniche_daten(self, canvas:object=None, x:int=None, y:int=None, techniche_daten_object:dict=None, font_size:int=None, line_break:int=None):
        start_x = x
        start_y = y

        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, 'Techniche Daten:')
        start_y = next_line(start_y, line_break)

        canvas.setFont(self.font, font_size)
        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(techniche_daten_object)
        else:
            shuffled_dict = [(key,val) for key,val in techniche_daten_object.items()]
        
        start_x_temp = start_x
        start_y_temp = start_y
        all_ys = []
        for i, (key, val) in enumerate(shuffled_dict):
            if val == 'N.A':
                val = ''
            canvas.drawString(start_x_temp, start_y_temp, f'{key}: {val}')
            start_y_temp = next_line(start_y_temp, line_break)
            all_ys.append(start_y_temp)
            if (i+1)%3 == 0:
                if len(shuffled_dict) > 6:
                    start_x_temp+=random.choice(np.arange(170,190))
                else:
                    start_x_temp+=250
                start_y_temp = start_y
        
        return start_x, min(all_ys)
    
    def draw_prufgrundlage(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, font_size:int=None, prufgrundlage_object:dict=None):
        start_x = x
        start_y = y
        canvas.setFont(self.font_bold, font_size+2)
        header = random.choice([f"Wiederkehrende Prüfung gemäß § {random.choice([14,15,16])} BetrSichV", 
                              f"Zwischenprüfung gemäß § {random.choice([14,15,16])} BetrSichV"])
        canvas.drawString(start_x, start_y, header)

        start_y = next_line(start_y, line_break * 2)
        key_val = list(prufgrundlage_object.items())
        
        key = key_val[0][0]
        val = key_val[0][1]

        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, f'{key}:')
        canvas.setFont(self.font, font_size)

        if len(key) < 14 : 
            canvas.drawString(start_x+80, start_y, val)
        else:
            canvas.drawString(start_x+100, start_y, val)

        return start_x, start_y

    def draw_festgestellte_mangel(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, font_size:int=None, festgestellte_mangel_object:dict=None):
        start_x = x
        start_y = y

        key_val = list(festgestellte_mangel_object.items())
        key = key_val[0][0]
        val = key_val[0][1]
        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, f'{key}:')
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)

        for i, mangels in enumerate(val):
            if mangels.lower() == 'keine':
                canvas.drawString(start_x, start_y, mangels)
                start_y = next_line(start_y, line_break)
            else:
                if mangels[0] in string.digits:
                    line1, line2= break_string(mangels, 4)
                else:
                    line1, line2= break_string(mangels, 22)
                canvas.drawString(start_x, start_y, f'{i}. {line1}')
                start_y = next_line(start_y, line_break)
                lines = break_string_recursively(line2, 80)
                for line in lines:
                    canvas.drawString(start_x, start_y,f'{line}')
                    start_y = next_line(start_y, line_break)
                start_y = next_line(start_y, line_break)
        return start_x, start_y

    def draw_hinweise(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, font_size:int=None, hinweise_object:dict=None):
        start_x = x
        start_y = y

        key_val = list(hinweise_object.items())
        key = key_val[0][0]
        val = key_val[0][1]

        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, key)
        start_y = next_line(start_y, line_break)
        canvas.setFont(self.font, font_size)
        canvas.drawString(start_x, start_y, val)

        return start_x, start_y

    def draw_prufergebnis(self, canvas:object=None, x:int=None, y:int=None, line_break:int=None, 
                                  font_size:int=None, prufergebnis_object:dict=None):
        start_x = x
        start_y = y
        
        key_val = list(prufergebnis_object.items())
        key = key_val[0][0]
        val = key_val[0][1]

        canvas.setFont(self.font_bold, font_size)
        canvas.drawString(start_x, start_y, f'{key}: {val}')

        return start_x, start_y

    def draw_pruf_interval(self, canvas:object=None, x:int=None, y:int=None, pruf_interval_object:list=None, key_break:int=None, font_size:int=None, line_break:int=None):
        start_x = x
        start_y = y

        canvas.setFont(self.font, font_size)

        if self.shuffle_dict:
            pruf_interval_object = {key_val[0]:key_val[1] for key_val in pruf_interval_object}
            shuffled_dict = shuffle_dict(pruf_interval_object)
        else:
            shuffled_dict = pruf_interval_object

        start_x_temp = start_x
        for i, (key,val) in enumerate(shuffled_dict):
            canvas.drawString(start_x_temp, start_y, f'{key}: {val}')
            start_x_temp+=key_break

        start_y = next_line(start_y, line_break*2)
        add_desc = random.choice([0,1,2])
        if add_desc>0:
            for i in range(add_desc):
                canvas.drawString(start_x, start_y, self.fixed_sentances['prufergebnis_footer'][i])
                if i < add_desc:
                    start_y = next_line(start_y, line_break)

        return start_x, start_y
    
    def draw_prufort_details(self, canvas:object=None, x:int=None, y:int=None, key_break:int=None, line_break:int=None, font_size:int=None, prufort_obect:dict=None):
        start_x =x
        start_y =y

        if self.shuffle_dict:
            shuffled_dict = shuffle_dict(prufort_obect)
        else:
            shuffled_dict = [(key,val) for key,val in prufort_obect.items()]

        canvas.setFont(self.font, font_size)
        start_x_temp = start_x

        for i, (key,val) in enumerate(shuffled_dict):
            canvas.drawString(start_x_temp, start_y, f'{key}: {val}')
            start_x_temp+= key_break
        
        start_y = next_line(start_y, 2*line_break)
        add_desc = random.choice([*[0 for _ in range(5)], *[1 for _ in range(80)], *[2 for _ in range(15)]])
        if add_desc == 1:
            canvas.drawString(start_x, start_y, self.fixed_sentances['prufort_desc'][0])
        elif add_desc ==2:
            canvas.drawString(start_x, start_y, self.fixed_sentances['prufort_desc'][1])
                

        return start_x, start_y

    def filter_keys(self, global_keys_key:dict=None, global_keys_no_key:dict=None, pruf_interval_object:list=None, techniche_date_type:list=None, key_mappings:dict=None):
        filtered_dict= {}

        keys_3c = [keys for keys in key_mappings['techniche_daten_3c']]
        keys_2c = [keys for keys in key_mappings['techniche_daten_2c']]
        keys_2c_all = [keys for keys in key_mappings['techniche_daten_2c_all']]
        keys_2c_pop = set(keys_2c_all) - set(keys_2c) 
        keys_3c_2c_pop = set(keys_3c) - set(keys_2c)
        keys_2c_3c_pop = set(keys_2c_all) - set(keys_3c)

        
        if len(techniche_date_type) == 6:
            [self.keys_to_include.pop(self.keys_to_include.index(key)) for key in keys_3c_2c_pop if key in self.keys_to_include]
            
            [self.keys_to_include.pop(self.keys_to_include.index(key)) for key in keys_2c_pop if key in self.keys_to_include]
        elif len(techniche_date_type) == 8:
            [self.keys_to_include.pop(self.keys_to_include.index(key)) for key in keys_2c_3c_pop if key in self.keys_to_include]
            
        
        for key,val in global_keys_key.items():
            if len(pruf_interval_object)>0:
                keys_pi = [key_val[0] for key_val in pruf_interval_object] # pruf interval object
                
                keys_pi_ki =  [key for key in self.keys_to_include if key in keys_pi] # keys in keys to include if pruf interval keys are mentioned 
                
                keys_to_pop= set(keys_pi_ki) - set(keys_pi)
                if len(list(keys_to_pop)) >0:
                    self.keys_to_include = [self.keys_to_include.pop(self.keys_to_include.index(key)) for key in keys_to_pop if key in self.keys_to_include]
            
            if key in self.keys_to_include:
                filtered_dict[key] = global_keys_key[key]
                if val == 'N.A':
                    self.keys_to_include.pop(self.keys_to_include.index(key))

        for key,val in global_keys_no_key.items():
            if key in self.keys_to_include:
                filtered_dict[key] = global_keys_no_key[key]
                if val == 'N.A':
                    self.keys_to_include.pop(self.keys_to_include.index(key))
                
        
        return filtered_dict
 


    def draw_report(self, report_name: str = 'form_tuv_nord.pdf', image_path = 'form_tuv_nord.jpg'):
        c = Canvas(report_name)
        canvas = c
        self.select_random_font(self.fonts_dir)
        canvas.setFont(self.font, 10)
        canvas.setPageSize(letter)
        canvas.setLineWidth(.5)
        start_x = self.start_x
        start_y = self.start_y
        
        key_mappings = self.init_key_mappings()
        global_keys_key, global_keys_no_key = self.init_global_keys()
        
        global_keys_key, global_keys_no_key = self.populate_global_keys(global_keys_key=global_keys_key, global_keys_no_key=global_keys_no_key)
        

        ## draw header
        
        start_x, new_line = self.draw_header(canvas=canvas, x=self.start_x, y = self.start_y, header_str=global_keys_no_key["Anschrift des Bewerters"], font_size=5, line_break=8)

        ## Draw owner address
        new_line = next_line(start_y, random.choice(np.arange(35,45, 2)))
        #start_y_temp = new_line # For prufbescheinigung
        _, new_line = self.draw_owner_address(canvas=canvas, x = start_x, y = new_line, font_size=9, line_break=10, owner_address=global_keys_no_key["Adresse des Besitzers"])


        ## Draw Prufbescheingung
        start_x_temp = random.choice(np.arange(260, 280, 5))
        prufbescheinigung_object = extract_object(global_object=global_keys_key, filter_keys=key_mappings['prufbescheinigung_object'])
        start_x_temp, new_line = self.draw_Prüfbescheinigung(canvas=canvas, x=start_x_temp, y=start_y, line_break=12, rect_height=75, rect_width=250, 
                                    font_size=8, prufbescheinigung_object=prufbescheinigung_object)

        ## Draw Prufbescheingung footer
        new_line = next_line(new_line, 12)
        #prufbescheinigung_object_footer = {key:val for key,val in global_keys_key.items() if key in key_mappings['prufbescheinigung_object_footer']}
        prufbescheinigung_object_footer = extract_object(global_object=global_keys_key, filter_keys=key_mappings['prufbescheinigung_object_footer'])
        _, new_line = self.draw_Prüfbescheinigung_footer(canvas=canvas, x=start_x_temp, y=new_line, line_break=8, footer_object=prufbescheinigung_object_footer, font_size=6)


        ## Draw personaufzug
        new_line = next_line(new_line, 18)
        #personeaufzug_object  = {key:val for key,val in global_keys_key.items() if key in key_mappings['personaufzug_object']}
        personeaufzug_object = extract_object(global_object=global_keys_key, filter_keys=key_mappings['personaufzug_object'])
        _, new_line = self.draw_personeaufzug(canvas=canvas, x=start_x, y= new_line, line_break=12 ,personeafuzug_object= personeaufzug_object, font_size=9)


        ## Draw Technische Daten
        new_line = next_line(new_line, 28)
        techniche_dict = random.choice([key_mappings['techniche_daten_2c'], key_mappings['techniche_daten_3c']])
        #techniche_daten_object = {key:val for key,val in global_keys_key.items() if key in techniche_dict}
        techniche_daten_object = extract_object(global_object=global_keys_key, filter_keys=techniche_dict)
        _, new_line = self.draw_techniche_daten(canvas=canvas, x=start_x, y=new_line, line_break=12, techniche_daten_object=techniche_daten_object,font_size=9)

        ## Draw prufgrundlage object

        new_line = next_line(new_line, 33)
        prufgrundlage_object = extract_object(global_object=global_keys_key, filter_keys=key_mappings['prüfgrundlage'])
        _, new_line = self.draw_prufgrundlage(canvas=canvas, x=start_x, y=new_line, font_size=9, prufgrundlage_object= prufgrundlage_object, line_break=12)

        ## Draw festgestellte mange
        new_line = next_line(new_line, 23)
        festgestellte_mange_object = extract_object(global_object=global_keys_key, filter_keys=key_mappings['festgestellte_mangel'])
        _, new_line = self.draw_festgestellte_mangel(canvas=canvas, x=start_x, y=new_line, line_break=12, font_size=9, festgestellte_mangel_object=festgestellte_mange_object)

        ## Draw hinweise
        new_line = next_line(new_line, 5)
        hinweise_object = extract_object(global_object=global_keys_key, filter_keys=['Erläuterungen/Hinweise'])
        _, new_line = self.draw_hinweise(canvas=canvas, x=start_x, y = new_line, line_break=12, font_size=9, hinweise_object=hinweise_object)
        
        ## Draw prufergebnis
        new_line = next_line(new_line, 18)
        prufergebnis_object = extract_object(global_object=global_keys_key, filter_keys=['Prüfergebnis'])
        _, new_line = self.draw_prufergebnis(canvas=canvas, x=start_x, y=new_line, font_size=9, prufergebnis_object=prufergebnis_object)

        ## Draw prufinterval
        pruf_interval_object = self.generate_pruf_interval_object(global_keys_key=global_keys_key, key_mapping=key_mappings['prufergebnis_interval'])
        if len(pruf_interval_object) > 0:
            new_line = next_line(new_line, 18)
            _, new_line = self.draw_pruf_interval(canvas=canvas, x=start_x, y= new_line, key_break=150, font_size=9, pruf_interval_object=pruf_interval_object, line_break=12)


        ##Draw prufort details
        new_line = next_line(new_line, 22)
        prufort_object = extract_object(global_object=global_keys_key, filter_keys=key_mappings['prufort_object'])
        
        _, new_line = self.draw_prufort_details(canvas=canvas, x=start_x, y=new_line, line_break=12, key_break=160, font_size=9, prufort_obect=prufort_object)


        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(image_path, 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        image = cv2.imread(image_path)[:,:,::-1]
        #global_keys_ext = {'global_keys':global_keys, 'global_keys_config':global_keys_config}
        filtered_dict = self.filter_keys(global_keys_key=global_keys_key, global_keys_no_key= global_keys_no_key, 
                                         pruf_interval_object=pruf_interval_object, techniche_date_type=techniche_dict, key_mappings=key_mappings)
        #print(filtered_dict)
        global_keys_ext = {'global_keys':filtered_dict, 'global_keys_config':{'font_size': 9}}
        #print(global_keys_ext)
        return global_keys_ext, image


if __name__=='__main__':
    start_x = 60
    start_y = 700
    fonts_dir = "fonts"
    sentance_gen_type='fixed'
    keys_to_include = ["Anlagen-Nr", "Standort", "Auftrags-Nr", "Service Techniker"]
    random_numbers = False
    sentance_dir = 'sentances_tuv_nord/fixed_sentances.json'
    
    for i in range(30):
        template_kone = Template_TUV_Nord(start_x=start_x, start_y = start_y, fonts_dir=fonts_dir, 
                                      keys_to_include = keys_to_include,  sentance_gen_type = sentance_gen_type, sentances_dir=sentance_dir, file_name='form_schindler.pdf', 
                                      random_numbers = random_numbers)
        template_kone.draw_report()
    #print(image.shape)
    #_,_,image = get_ocr_data(image = image)
    #print(type(image))
    #cv2.imwrite('schinlder_ocr.png', image)