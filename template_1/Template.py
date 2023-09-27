import random
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

class Template:
    def __init__(self):
        pass
    
    def draw_report(self, report_name:str=None, image_path:str = None):
        pass
    
    def generate_ocr(self):
        pass

    def select_random_font(self, font_dir:str):
        font = random.choice([f for f in os.listdir(font_dir) if 'bd' not in f[:-4]])
        font = font[:-4]
        self.font =  font
        self.font_bold = f'{self.font}-bold'
        pdfmetrics.registerFont(TTFont(f'{self.font}', f'{font_dir}/{font}.ttf'))
        pdfmetrics.registerFont(TTFont(f'{self.font_bold}', f'{font_dir}/{font}bd.ttf'))
    
    