import os


from Template import Template

import json
#pwd = os.getcwd()
#os.chdir("..")
from utility_functions.utilities_kie import *
#os.chdir(pwd)
from faker import Faker
locales = ['de_DE']
Faker.seed(0)

#pytesseract.pytesseract.tesseract_cmd = r'C://Program Files//Tesseract-OCR//tesseract.exe'
pdfmetrics.registerFont(TTFont('Arial', './template_1/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', './template_1/arialbd.ttf'))

header = 'Zugelassene Überwachungsstelle Aufzüge'
file_name = '20181119-32753-1891960176-100-421500.docx'
page_no = 'Seite 1 von 1'


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
        self.synonyms_file = json.load(open("./template_1/synonyms_test.json"))
        self.image_name = report_name[:-4]+'.jpg'
        self.page_no = page_no
        
    def init_utililty_info(self):
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
        return utility_info
        
    def init_global_keys(self):
        utility_info = self.init_utililty_info()
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
            'Förderhöhe' : '93,68 m', #distance in meters
            'Ergebnis der Prüfung' : 'Keine Mängel' 
        },

        'technical_specifications_config': {
        'font_size' : 10,
        'font-type-keys' : 'Arial-Bold',
        'font-type-vals' : 'Arial',
        'line-break' : 20,
        'key-val-spacing' : 140
    
        }}
        return global_keys
    
    def next_line(self, start_y: int, line_break: int):
        return start_y - line_break
    
    def rearange_key_vals_test_results(self, dict_list : dict, keys_to_keep: list): ## keys which needs to appear either of the positions mentioned in indices usually 1,6
        already_swapped_indices = []
        swap_indices = np.arange(0,5)
        keys = [key for i, (key, val) in enumerate(dict_list)]
        #print(keys)
        '''for i , (key, val) in enumerate(dict_list):
            if key in keys_to_keep:
                if i >6:
                    while True:
                        print("here")
                        swap_index = random.choice(swap_indices)
                        if dict_list[swap_index] not in keys_to_keep:
                            dict_list[swap_index], dict_list[i] = dict_list[i], dict_list[swap_index]
                            break'''
        for key in keys_to_keep:
            if keys.index(key) >= 6:
                i = keys.index(key)
                while True:
                    #print("here")
                    swap_index = random.choice([ i for i in swap_indices if i not in (1,2)])
                    if dict_list[swap_index][0] not in keys_to_keep:
                        dict_list[swap_index], dict_list[i] = dict_list[i], dict_list[swap_index]
                        break
                        
                        
        for i , (key, val) in enumerate(dict_list):
            if key in keys_to_keep and i >= 6:      
                print(f'index {dict_list[i]}:{i}')

        return dict_list
    
    def generate_company_description(self):
        fake = Faker(locales)
        return fake.bs().replace('\n', ' ')
    
    def generate_person_name(self):
        fake = Faker(locales)
        return f'{fake.prefix()} {fake.name()}'.replace('\n', '')
    
    def generate_telephone_number(self):
        fake = Faker(locales)
        area_code = fake.random_int(min=2, max=99)
        subscriber_number = fake.random_number(digits=5)
        suffix_number = random.choice([fake.random_number(digits=3), fake.random_number(digits=1)])
        telephone_number = f"+49.{area_code}.{subscriber_number}-{suffix_number}"
        return telephone_number.replace('\n', '')
    
    def generate_person_email(self):
        fake = Faker(locales)
        return fake.ascii_company_email().replace('\n', '')
        
    def generate_evaluator_adrress(self, utility_info:dict = None):
        fake = Faker(locales)
        company_name = self.generate_random_company()
        company_desc = self.generate_company_description()
        comapny_address_street = fake.street_name().replace('\n', '') #line 1
        comapny_address_building_no = fake.random_int(min=1, max = 1000) #line 
        company_address_pzl = fake.postcode().replace('\n', '') # line 3
        company_address_city = fake.city().replace('\n', '') # line3
        
        telephone = self.generate_telephone_number()
        fax = self.generate_telephone_number()
        name = self.generate_person_name()
        person_phone = self.generate_telephone_number()
        person_email = self.generate_person_email()

        
        if utility_info is None:
            utility_info = self.init_utililty_info()
        
        utility_info["evaluator_address"]["address"]["line1"] = company_name
        utility_info["evaluator_address"]["address"]["line2"] = company_desc
        utility_info["evaluator_address"]["address"]["line3"] = comapny_address_street
        utility_info["evaluator_address"]["address"]["line4"] = comapny_address_building_no
        utility_info["evaluator_address"]["address"]["line5"] = company_address_pzl + ' ' + company_address_city
        utility_info["evaluator_address"]["Telefon"] = telephone
        utility_info["evaluator_address"]["Telefax"] = fax
        utility_info["evaluator_address"]["Kontakt"]["line1"] = name
        utility_info["evaluator_address"]["Kontakt"]["line2"] = f'Tel. direkt {person_phone}'
        utility_info["evaluator_address"]["Kontakt"]["line3"] = f'E-Mail {person_email}'
        
        
        return utility_info
        
    
    def generate_client_address(self, utility_info:dict = None, line_num:int=4):
        fake = Faker(locales)
        company_name = self.generate_random_company()
        #company_name_2 = f' c/o {self.generate_random_company()}'
        company_desc = self.generate_company_description()
        comapny_address_street = fake.street_name().replace('\n', '') #line 1
        comapny_address_building_no = fake.random_int(min=1, max = 1000) #line 
        company_address_pzl = fake.postcode().replace('\n', '') # line 3
        company_address_city = fake.city().replace('\n', '') # line3
        
        if line_num == 3:
            utility_info["client_address"]["line1"] = company_name
            utility_info["client_address"]["line2"] = f'{comapny_address_street} {comapny_address_building_no}'
            utility_info["client_address"]["line3"] = f'{company_address_pzl} {company_address_city}'
            utility_info["client_address"].pop("line4", None)
        else:
            utility_info["client_address"]["line1"] = company_name
            utility_info["client_address"]["line2"] = company_desc
            utility_info["client_address"]["line3"] = f'{comapny_address_street} {comapny_address_building_no}'
            utility_info["client_address"]["line4"] = f'{company_address_pzl} {company_address_city}'
        return utility_info
        
    
    def generate_random_address(self):
        fake = Faker(locales)
        return fake.address().replace('\n', ' ')
    
    def generate_random_company(self):
        fake = Faker(locales)
        return fake.company().replace('\n', ' ')
    
    def generate_eigennummer(self):
        part1 = ''.join([random.choice(string.ascii_uppercase) for _ in range(0,4)])
        part1 = part1[:2] + '-' + part1[2:]
        part2 = ''.join([str(random.choice(string.digits)) for _ in range(0,5)])
        eigennummer = part1 + ' ' + part2
        return eigennummer
    
    def generate_messegrate(self):
        messegrate_divices = ['0100', 'MXTRA', 'MTECH', '0100s ii', 'pv 1500']
        return f'Profitest {random.choice(messegrate_divices)}'
    
    def generate_eigename(self):
        part1 = ''.join([random.choice(string.ascii_uppercase) for _ in range(0,2)])
        part2 = ''.join([random.choice(string.digits) for _ in range(0,4)])

        eigename = part1+ ' ' + part2
        return eigename
    
    def generate_baujahr(self):
        baujahren = np.arange(1999, 2023)
        baujahr = random.choice(baujahren)
        return str(baujahr)
    
    def generate_fabrik_nr(self):
        return ''.join([random.choice(string.digits) for _ in range(0,7)])
    
    def generate_antrieb(self):
        antrieb_types = ['Treibscheibe', 'Traktionsantrieb', 'Spindelantrieb', 'Zahnstangenantrieb']
        antrieb_ratios = ['2:1', '1:1', '4:1']
        
        return random.choice(antrieb_types)+' / '+random.choice(antrieb_ratios)
    
    def generate_errichtungsgrundlage(self):
        errichtungsgrundlage_list = ['EN 81-1: 1998+A3: 2009', 'EN 81-20', 'EN 81-1: 1998', 'TRA 200']
        return random.choice(errichtungsgrundlage_list)
    
    def generate_tragfähigkeit(self):
        
        kgs = np.arange(300,1500, 50)
        
        start = 2
        kgs_pers = {kgs[i]:start+i+1 for i in range(len(kgs))}
        
        key = random.choice(list(kgs_pers.keys()))
        
        return f'{key} kg / {kgs_pers[key]} Pers'
    
    def generate_haltstellen(self):
        haltstellen = random.choice(np.arange(2,30))
        haltstellen = f'{haltstellen} / {haltstellen}'
        return haltstellen
    
    def populate_test_certificate_results_fake(self, unified_dict:dict= None, utility_info : dict = None):
        address_objektstandort = self.generate_random_address()
        unified_dict['test_certificate_results']['Objektstandort'] = address_objektstandort
        word = unified_dict['test_certificate_results']['Objektart / Anlage']
        
        unified_dict['test_certificate_results']['Objektart / Anlage'] = random.choice(self.synonyms_file[word])
        unified_dict['test_certificate_results']['Arbeitgeber'] = ' '.join([utility_info["client_address"][key] for key in utility_info["client_address"].keys()])
        unified_dict['test_certificate_results']['Eigennummer'] = self.generate_eigennummer()
        unified_dict['test_certificate_results']['Eigenname'] = self.generate_eigename()
        unified_dict['test_certificate_results']['Fabrik-Nr.'] = self.generate_fabrik_nr()
        unified_dict['test_certificate_results']['Verwendete Messgeräte'] = self.generate_messegrate()

        return unified_dict
    
    def populate_technical_specifications_fake(self, unified_dict:dict=None):
        address = self.generate_random_address()
        unified_dict["technical_specifications"]["Baujahr"] = self.generate_baujahr()
        unified_dict["technical_specifications"]["Wartungsfirma"] = address
        unified_dict["technical_specifications"]['Errichtungsgrundlage'] = self.generate_errichtungsgrundlage()
        unified_dict["technical_specifications"]['Antrieb / Aufhängung'] = self.generate_antrieb()
        unified_dict["technical_specifications"]['Hersteller'] = address
        unified_dict["technical_specifications"]['Tragfähigkeit'] = self.generate_tragfähigkeit()
        unified_dict["technical_specifications"]['Haltestellen / Zugangstellen'] = self.generate_haltstellen()
        
        return unified_dict
        
    
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
        bold_flag = 0
        count =0
        for key in evaluator_address.keys():
            if not type(evaluator_address[key]) == dict: 
                canvas.drawString(start_x_temp, y, str(key) + ' : ' + str(evaluator_address[key]))
                count += 1
                y = self.next_line(y, evaluator_address_config['line_break'])
            else:
                for i,k in enumerate(evaluator_address[key].keys()):
                    if i==0 and bold_flag == 0:
                        bold_flag = 1
                        canvas.setFont(f'{evaluator_address_config["font"]}-Bold', evaluator_address_config['font_size'])
                    else:
                        canvas.setFont(evaluator_address_config['font'], evaluator_address_config['font_size'])
                    canvas.drawString(start_x_temp, y, str(evaluator_address[key][k]))
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
        items_test_certificate_results = self.rearange_key_vals_test_results(items_test_certificate_results, test_certificate_results_config['vertical-left-only'])
        
        
        start_x_temp = x
        new_lines = []
        for i , (key, val) in enumerate(items_test_certificate_results):
            start_x_temp_old = None
            y_temp = None
            if i == 6:
                start_x_temp = start_x_temp + 2 * test_certificate_results_config['key-val-spacing']
                y = new_lines[-5]
                
            
            canvas.setFont(test_certificate_results_config['font-type-keys'], test_certificate_results_config['font_size'])
            canvas.drawString(start_x_temp, y , key)
            #start_x_temp = start_x_temp + test_certificate_results_config['key-val-spacing']
            
            canvas.setFont(test_certificate_results_config['font-type-vals'], test_certificate_results_config['font_size'])
            if len(str(val).split(" "))> 3 or len(str(val).split(" ")[0]) >= 20:
                
                if len(str(val).split(" ")[0]) >= 20:
                    split_index = 1
                else:
                    split_index = 3    
                canvas.drawString(start_x_temp + test_certificate_results_config['key-val-spacing'], y, ' '.join(str(val).split(" ")[:split_index]))
                y = self.next_line(y, 12)
                canvas.drawString(start_x_temp + test_certificate_results_config['key-val-spacing'], y, ' '.join(str(val).split(" ")[split_index:]))
                if(len(str(val).split(" ")[split_index:]) > 3):
                    y = self.next_line(y, 8)
            else:
                canvas.drawString(start_x_temp + test_certificate_results_config['key-val-spacing'], y, str(val))
            
            y = self.next_line(y, test_certificate_results_config['line-break'])
            new_lines.append(y)

        return x, min(new_lines)
     
    def draw_technical_specifications(self, technical_specifications, technical_specifications_config, canvas, x, y):
        start_x_temp = x
        start_y_temp = y
        shuffled_dict_tuple = self.shuffle_dict(technical_specifications)
        new_lines = []
        for i, (key, val) in enumerate(shuffled_dict_tuple):
            if i == 4:
                start_x_temp += 2*technical_specifications_config['key-val-spacing']
                start_y_temp = y
            
            canvas.setFont(technical_specifications_config['font-type-keys'], technical_specifications_config['font_size'])
            canvas.drawString(start_x_temp, start_y_temp, str(key))
            canvas.setFont(technical_specifications_config['font-type-vals'], technical_specifications_config['font_size'])
            #start_x_temp += technical_specifications_config['key-val-spacing']
            
            if len(str(val).split(" ")) >= 3: 
                str_eval = ' '.join(str(val).split(" ")[:2])
                #print(f'{val}')
                split_index = 1 if len(str_eval) >= 22 else 3
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, ' '.join(str(val).split(" ")[:split_index]))
                start_y_temp = self.next_line(start_y_temp, 12)
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, ' '.join(str(val).split(" ")[split_index:]))
                if(len(str(val).split(" ")[split_index:]) > 3):
                    start_y_temp = self.next_line(start_y_temp, 4)
            else:
                canvas.drawString(start_x_temp + technical_specifications_config['key-val-spacing'], start_y_temp, str(val))
            start_y_temp = self.next_line(start_y_temp, technical_specifications_config['line-break'])
            new_lines.append(y)
        return x, min(new_lines)

    def draw_report(self,header:dict=None, report_name:str='form.pdf', global_keys:dict = None):
        
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
        utility_info = self.init_utililty_info()
        line_num = random.choice([3,4])
        utility_info = self.generate_client_address(utility_info=utility_info, line_num=line_num)
        
        _, new_line = self.draw_client_address(utility_info['client_address'], utility_info['client_address_config'], canvas, self.start_x, new_line)

        #canvas.line(480,747,580,747)
  
        #Section 2
        utility_info = self.generate_evaluator_adrress(utility_info=utility_info)
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
        global_keys = self.populate_test_certificate_results_fake(unified_dict=global_keys, utility_info = utility_info)
        _, new_line = self.draw_test_certificate_results(global_keys['test_certificate_results'], global_keys['test_certificate_results_config'], canvas, self.start_x, new_line)
        new_line = self.next_line(new_line, self.line_break)
        
        ## Section 6
        canvas.setFont('Arial-Bold', 10)
        canvas.drawString(self.start_x, new_line - random.choice([-3, -4, -5]), 'Technische Angaben')
        new_line = self.next_line(new_line, self.line_break)
        global_keys = self.populate_technical_specifications_fake(unified_dict=global_keys)
        _, new_line = self.draw_technical_specifications(global_keys['technical_specifications'], global_keys['technical_specifications_config'], canvas, self.start_x, new_line)
        #_, new_line = self.draw_test_certificate_results(test_certificate_results, test_certificate_results_config, canvas, self.start_x, new_line)

        canvas.save()
        pages = convert_from_path(report_name, 500)
        pages[0].save(f'{report_name[:-4]}.jpg', 'JPEG')
        #plt.figure(figsize = (200,10))
        #plt.imshow(cv2.imread('form.jpg')[:,:,::-1])
        image = cv2.imread(f'{report_name[:-4]}.jpg')[:,:,::-1]
        return global_keys, image
    

        
            
        
        
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
    #template1.draw_report(header=header, report_name='form2.pdf')
    #pages = convert_from_path('form.pdf', 500)
    #pages[0].save(f'form.jpg', 'JPEG')
    model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    #print(f'len(tokenizer) {len(tokenizer)}')
    tokens, bboxes, image = template1.get_ocr_data(image_path='form2.jpg')

    #print([token for token in tokens if token in set(string.punctuation) and token in (' ', '')])
    #print('------------------------')
    #template1.add_tokens(tokens=tokens)
    #tokens = [token.lower() for token in tokens if token != ' ']
    tokens, bboxes = template1.preprocess_tokens(tokens=tokens, bboxes=bboxes)

    #tokens = np.load('tokens_temp.npy')
    #bboxes = np.load('bboxes_temp.npy')
    #print(f'len(tokens) {len(tokens)}')
    #print(f'len(bboxes) {len(bboxes)}')
    #plt.figure(figsize=(20,10))
    #plt.imshow(image)
    tokenizer, model = template1.add_tokens_tokenizer(tokens = tokens, tokenizer = tokenizer, model = model)
    input_ids, bboxes, input_id_map = template1.encode_tokens(tokens=tokens, bboxes=bboxes, tokenizer=tokenizer)
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
    labels= template1.label_input_ids(unified_dict=key_vals_unified,tokens = tokens,  bboxes = bboxes, input_ids=input_ids, input_id_map=input_id_map, tokenizer=tokenizer)
    #token_group_key, token_group_val, token_group_others = template1.form_token_groups(unified_dict=key_vals_unified, tokens=tokens, bboxes=bboxes)    
    key_set, val_set, token_map = template1.form_token_groups(unified_dict=key_vals_unified, tokens=tokens, bboxes=bboxes)
    entities = template1.form_entities(unified_dict=key_vals_unified,tokens = tokens,  bboxes = bboxes, input_ids=input_ids, input_id_map=input_id_map, tokenizer=tokenizer)
    
    #for id, label in zip(input_ids, labels):
    #    print(f'{tokenizer.decode(id)}, {label}')
    
    #print(tokenizer.tokenize('Prüfbescheinigung, Wiederkehrende Prüfung (Hauptprüfung)'.lower()))
    #for id in input_ids:
    #    print(f'{tokenizer.decode(id)}')
    #print(f' outside method lenght of input ids {len(input_ids)}') 
    #for id in input_ids:
    #    print(tokenizer.decode([id]))