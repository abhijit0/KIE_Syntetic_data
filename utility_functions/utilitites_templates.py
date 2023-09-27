import random
import string

def break_string(val, length):
    
    if len(val) < length:
        return val, '0'
    
    val_splits = val.split(" ")
    str = ''
    #length -= len(val_splits)
    if len(val_splits) < 2:
        return val, '0'
    for i in range(len(val_splits)):
        str += val_splits[i] + ' ' if i < len(val_splits) else val_splits[i]
        if len(str) >= length:
            if i >= 1:
                line_1 = ' '.join([val_splits[j] for j in range(i)])
                line_2 = ' '.join([val_splits[j] for j in range(i, len(val_splits))])
            elif i ==0:
                line_1 = ''.join(val_splits[0])
                line_2 = ' '.join(val_splits[1:])
                
            return line_1, line_2
    return '0', '0'

def break_string_recursively(string:str, length:int):
    lines = []
    string = string.replace("\n", " ")
    
    line_1, line_2 = break_string(string, length)
    
    if line_1 != '0':
        lines.append(line_1)
    while True:
        if line_1 == '0' or line_2 == '0':
            break
        if len(line_2) <= length and line_2 != '0':
            lines.append(line_2)
            break
        if len(line_2) > length:
            line_1, line_2 = break_string(line_2, length)
            if line_1 != '0':
                lines.append(line_1)

    return list(lines)

def next_line(start_y: int, line_break: int):
        return start_y - line_break
    
def shuffle_dict(dict_):
    dict_items = list(dict_.items())
    random.shuffle(dict_items)
    return dict_items

def generate_random_digit_string(len:int):
    add_zero_prefix = [i<20 for i in range(100)]
    if add_zero_prefix:
        return '0'+''.join([random.choice(string.digits) for _ in range(len-1)])
    else:
        first_digit = [str(i) for i in range(1,9)]
        return first_digit + ''.join([random.choice(string.digits) for _ in range(len-1)])

def extract_object(global_object:dict=None, filter_keys:list=None):
    filtered_object = {key:global_object[key] for key in filter_keys if key in global_object.keys()}
    return filtered_object

