
import os
import json
import sys

sys.path.append('./utility_functions/')
sys.path.append('./template_1/')
sys.path.append('./template_1/sentances_kone')
sys.path.append('./template_1/fonts')


from utility_functions.utilities_kie import *
from template_1.template1 import Template_Dekra
from template_1.template_2_kone import Template_Kone
from template_1.template_schindler import Template_Schindler
from template_1.template_tuv_nord import Template_TUV_Nord
import pickle
import shutil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import LayoutLMv2FeatureExtractor
import torch

from typing import Optional, Union

class DatasetGenerator:
    def __init__(self, num_files:int=None, images_dir:str=None, images_resized_dir:str=None, bbox_dir:str = None, input_ids_dir :str = None, 
                 labels_dir :str = None, entities_dir :str =None, relations_dir:str = None, type:str=None, clear_all_old_files:bool=None, clear_old_files_type:list=None, datasets_init_configs:dict=None):
        self.num_files = num_files
        self.images_dir = images_dir
        self.bbox_dir = bbox_dir
        self.images_resized_dir = images_resized_dir
        self.input_ids_dir = input_ids_dir
        self.labels_dir = labels_dir
        self.entities_dir = entities_dir
        self.relations_dir = relations_dir
        self.type = type
        self.clear_all_old_files = clear_all_old_files
        self.clear_old_files_type = clear_old_files_type
        self.datasets_init_configs = datasets_init_configs
        self.type_b = False  ## Only used while generating kone documents
        #assert (key.lower() for key in self.datasets_init_configs.keys()) in ('kone', 'dekra', 'schindler', 'tuv')
    
    def get_tuv_nord_doc(self, init_config=None):
        template_schindler = Template_TUV_Nord(**init_config)
        global_keys, image = template_schindler.draw_report()
        return global_keys, image

    def get_schindler_doc(self, init_config = None):
        template_schindler = Template_Schindler(**init_config)
        global_keys, image = template_schindler.draw_report()
        return global_keys, image
        
    def get_dekra_doc(self, init_config:dict=None):
      
        template1 = Template_Dekra(**init_config)
        
        global_keys = template1.init_global_keys()
        global_keys, image= template1.draw_report(header=init_config["header"], report_name=init_config["report_name"], global_keys=global_keys)
        return global_keys, image
    
    def get_kone_doc(self, init_config:dict=None):
        template_kone= Template_Kone(**init_config)
        if not self.type_b:
            template_kone.draw_type = '4c'
        global_keys, image = template_kone.draw_report()
        return global_keys, image
        
    def generate_sample(self, model:AutoModel=None, tokenizer:AutoTokenizer=None, template_configs:dict = None):
        
        global_keys, image = template_configs["get_doc_method"](init_config= template_configs["init_configs"])       
        tokens, bboxes, _ = get_ocr_data(image=image)
        #print(tokens)
        tokens, bboxes = preprocess_tokens(tokens=tokens, bboxes=bboxes)
        
        tokenizer, model = add_tokens_tokenizer(tokens = tokens, tokenizer = tokenizer, model = model)
        #print(f'tokenized result {tokenizer.tokenize("ergebnis")}')
        input_ids, bboxes, input_id_map = encode_tokens(tokens=tokens, bboxes=bboxes, tokenizer=tokenizer)
        #print(len(input_ids))
        #input_ids = sorted(input_ids)
        
        key_vals_unified = unify_keys_vals(global_keys)
        
        #key_vals_unified = global_keys
        key_set, val_set, token_map = form_token_groups(unified_dict=key_vals_unified, tokens=tokens, bboxes=bboxes)
        #print(key_set)
        #print(val_set)
        #print('--')
        labels= label_input_ids(key_set=key_set, val_set=val_set, input_id_map=input_id_map, tokenizer=tokenizer, input_ids=input_ids)
        entities, entity_key_index_mapping, entity_key_index_mapping_reverse = form_entities(unified_dict=key_vals_unified,tokens = tokens,  bboxes = bboxes, input_ids=input_ids, input_id_map=input_id_map, tokenizer=tokenizer)
        
        if entities ==0 or entity_key_index_mapping ==0 or entity_key_index_mapping==0:
            return 0, 0, 0, 0,0,0
        
        relations = form_relations(entities=entities, unified_dict=key_vals_unified, key_set=key_set,  entity_key_index_mapping=entity_key_index_mapping, entity_key_index_mapping_reverse = entity_key_index_mapping_reverse)
        #print(relations)
        if relations==0:
            return 0, 0, 0, 0,0,0
        return image, input_ids, bboxes, labels, entities, relations
    
    def dump_pickle(self, file_name:str=None, collection:object=None):
        file_name = os.path.join(os.getcwd(), file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(collection, f)
        
    def initialize_tokenizer_model(self, tokenizer_dir:str=None, model_dir:str=None):
        
        if tokenizer_dir is None and model_dir is None:
            model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
        else:
            if not os.path.exists(tokenizer_dir):
                tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
            if not os.path.exists(model_dir):
                
                model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
            if self.clear_all_old_files:
                if os.path.exists(tokenizer_dir) and os.path.exists(model_dir):
                    shutil.rmtree(tokenizer_dir)
                    shutil.rmtree(model_dir)
                    model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
                    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
            elif len(os.listdir(os.path.join(os.getcwd(), tokenizer_dir)))== 0 and len(os.listdir(os.path.join(os.getcwd(), model_dir)))== 0:
                model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
                tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
            else:
                model = LayoutLMv2ForRelationExtraction.from_pretrained(model_dir)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return tokenizer, model
    
    def get_doc_method_mapping(self, key:str=None):
        with open(self.datasets_init_configs[key]) as f:
            init_config = json.load(f)
        
        get_doc_method_map = {'kone': self.get_kone_doc, 'dekra': self.get_dekra_doc, 'schindler':self.get_schindler_doc, "tuv_nord":self.get_tuv_nord_doc}
        template_configs = {"get_doc_method":get_doc_method_map[key], "init_configs":init_config}
        return template_configs
    
    def generate_Dataset(self, target_dir:str='dataset/', tokenizer_dir:str=None, model_dir:str = None):
        sub_dirs = [self.images_dir, self.images_resized_dir, self.bbox_dir, self.input_ids_dir, self.labels_dir, self.entities_dir, self.relations_dir]
        
        tokenizer, model = self.initialize_tokenizer_model(tokenizer_dir=tokenizer_dir, model_dir=model_dir)
        
        if self.clear_all_old_files:
            shutil.rmtree(target_dir)
            os.mkdir(target_dir)
            
        if self.clear_old_files_type is not None and self.clear_old_files_type[0] and not self.clear_all_old_files:
            shutil.rmtree(f'{target_dir}/{self.clear_old_files_type[1]}')
        
        if not os.path.exists(f'{target_dir}/{self.type}'):
            os.mkdir(f'{target_dir}/{self.type}')
        
        for dir in sub_dirs:
            joined_path= os.path.join(f'{target_dir}/{self.type}', dir)
            if not os.path.exists(joined_path):
                os.mkdir(joined_path)
        
        
        count = 0
        type_b_index = False
        for (key, val) in self.datasets_init_configs.items():
            print(f'Generating dataset for {key}')       
            for i in tqdm(range(self.num_files)):
                
                image = 0
                while type(image) is int:
                    template_config = self.get_doc_method_mapping(key=key)
                    
                    if key == 'kone' and i>=self.num_files//2:
                        self.type_b = True
                    
                    image, input_ids, bboxes, labels, entities, relations = self.generate_sample(model=model, tokenizer=tokenizer, template_configs=template_config)
                
                    if type(image) is not int or type(bboxes) is not int or type(input_ids) is not int or type(labels) is not int or type(entities) is not int or type(relations) is not int:
                        #print('Invalid OCR extraction skipping the sample')
                        #continue
                        image_resized = cv2.resize(image, (224,224))
                        image_name= os.path.join(f'{target_dir}/{self.type}/{self.images_dir}/image_{count + i}.jpeg')
                        images_resized_name= os.path.join(f'{target_dir}/{self.type}/{self.images_resized_dir}/image_{count +i}.jpeg')
            
                        cv2.imwrite(image_name, image)
                        cv2.imwrite(images_resized_name, image_resized)
                        ## input_ids
                    
                        bbox_file_name = os.path.join(f'{target_dir}/{self.type}/{self.bbox_dir}/bbox_{count + i}.p')
                        self.dump_pickle(file_name=bbox_file_name, collection=bboxes)
                    
                        input_ids_file_name = os.path.join(f'{target_dir}/{self.type}/{self.input_ids_dir}/input_ids_{count + i}.p')
            
                        self.dump_pickle(file_name=input_ids_file_name, collection=input_ids)
            
                        ## entities
                        entities_file_name = os.path.join(f'{target_dir}/{self.type}/{self.entities_dir}/entities_{count+i}.p')
            
                        self.dump_pickle(file_name=entities_file_name, collection=entities)
            
                        ## labels
                        labels_file_name = os.path.join(f'{target_dir}/{self.type}/{self.labels_dir}/labels_{count + i}.p')
            
                        self.dump_pickle(file_name=labels_file_name, collection=labels)
            
                        ## relations
                        relations_file_name = os.path.join(f'{target_dir}/{self.type}/{self.relations_dir}/relations_{count +i}.p')
            
                        self.dump_pickle(file_name=relations_file_name, collection=relations)
            count += self.num_files
        
        tokenizer.save_pretrained('./tokenizer_added_tokens')
        model.save_pretrained('./model_added_tokens')
    
    def path_exist(self, path:str=None):
        return os.path.exists(path)    
    
    def test_gnerator(self, tokenizer_path:str=None, model_path:str=None, file_index:int=None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = LayoutLMv2ForRelationExtraction.from_pretrained(model_path)
        image_path = f'dataset/train/images/image_{file_index}.jpeg'
        image_resized_path = f'dataset/train/images_resized/image_{file_index}.jpeg'
        
        #print(os.listdir(f'dataset/{self.type}/input_ids'))
        input_id_path = f'dataset/{self.type}/input_ids/input_ids_{file_index}.p'
        bbox_path = f'dataset/{self.type}/bbox/bbox_{file_index}.p'
        labels_path = f'dataset/{self.type}/labels/labels_{file_index}.p'
        entities_path = f'dataset/{self.type}/entities/entities_{file_index}.p'
        relations_path = f'dataset/{self.type}/relations/relations_{file_index}.p'
        
        entities = None
        if self.path_exist(path = image_path):
            image = cv2.imread(image_path)
            
        if self.path_exist(path = image_resized_path):
            image_resized = cv2.imread(image_resized_path)

        if self.path_exist(path = input_id_path):
            with open(input_id_path, 'rb') as f:
                input_ids = pickle.load(f)
        
        if self.path_exist(path = bbox_path):
            with open(bbox_path, 'rb') as f:
                bbox = pickle.load(f)

        if self.path_exist(path = labels_path):
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)

        if self.path_exist(path = entities_path):
            with open(entities_path, 'rb') as f:
                entities = pickle.load(f)
        if self.path_exist(path = relations_path):
            with open(relations_path, 'rb') as f:
                relations = pickle.load(f)
       
        if entities is not None:
            entity_names = []
            #print(entities)
            for i , (start, end) in enumerate(zip(entities['start'], entities['end'])):
                entity_names.append(tokenizer.decode(input_ids[start:end]))
            #print(entity_names)
            for i in range(len(relations['head'])):
                print(f'question : {entity_names[relations["head"][i]]}, Answer :  {entity_names[relations["tail"][i]]}, start: {relations["head"][i]}, end: {relations["tail"][i]}')

        else:
            print('not loaded properly')
        
            
class Custom_Dataset(Dataset):
    def __init__(self, images_dir:str=None, images_resized_dir:str=None, bbox_dir:str = None, input_ids_dir :str = None, 
                 labels_dir :str = None, entities_dir :str =None, relations_dir:str = None, type:str=None):
        
        
        self.type=type
        self.images_dir = f'dataset/{self.type}/{images_dir}'
        self.images_resized_dir = f'dataset/{self.type}/{images_resized_dir}'
        self.bbox_dir = f'dataset/{self.type}/{bbox_dir}'
        self.input_ids_dir = f'dataset/{self.type}/{input_ids_dir}'
        self.labels_dir = f'dataset/{self.type}/{labels_dir}'
        self.entities_dir = f'dataset/{self.type}/{entities_dir}'
        self.relations_dir = f'dataset/{self.type}/{relations_dir}'
        
    
    def generate_id(self, idx:int = None):
        return f'sample_{idx}'
    
    def __len__(self):
        return len(os.listdir(self.images_dir))
    
    def get_file_index_map(self, path:str=None):
        paths = os.listdir(path)
        sorted_paths = sorted(paths)
        idx_map = {i:sorted_paths[i] for i in range(len(sorted_paths))}
        return idx_map
    
    def __getitem__(self, idx):
        idx_map_image = self.get_file_index_map(path = self.images_dir)
        idx_map_image_resized = self.get_file_index_map(path = self.images_resized_dir)
        idx_map_bbox = self.get_file_index_map(path = self.bbox_dir)
        idx_map_labels = self.get_file_index_map(path = self.labels_dir)
        idx_map_entities = self.get_file_index_map(path = self.entities_dir)
        idx_map_realations = self.get_file_index_map(path = self.relations_dir)
        idx_map_input_ids = self.get_file_index_map(path = self.input_ids_dir)

        image = cv2.imread(f'{self.images_dir}/{idx_map_image[idx]}')
        image_resized = cv2.imread(f'{self.images_resized_dir}/{idx_map_image_resized[idx]}')
        
        bbox_path = f'{self.bbox_dir}/{idx_map_bbox[idx]}'
        with open(bbox_path, 'rb') as f:
            bbox = pickle.load(f)
            
        input_ids_path = f'{self.input_ids_dir}/{idx_map_input_ids[idx]}'
        with open(input_ids_path, 'rb') as f:
            input_ids = pickle.load(f)
            
        labels_path = f'{self.labels_dir}/{idx_map_labels[idx]}'
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        
        entities_path = f'{self.entities_dir}/{idx_map_entities[idx]}'
        with open(entities_path, 'rb') as f:
            entities = pickle.load(f)
        
        realtions_path = f'{self.relations_dir}/{idx_map_realations[idx]}'
        with open(realtions_path, 'rb') as f:
            relations = pickle.load(f)
        
        id = self.generate_id(idx)
        features  = {'id': id, 'input_ids': input_ids, 'bbox':bbox, 'labels':labels, 'original_image': image, 'image':image_resized, 'entities':entities, 'relations':relations}
        '''Dataset = {
            "features" : {'id': id, 'input_ids': input_ids, 'labels':labels, 'original_image': image, 'image':image_resized, 'entities':entities, 'relations':relations},
            "num_rows" : self.__len__()    
        }'''
        return features
         

@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values

        # prepare text input
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["original_image"]
            del feature["entities"]
            del feature["relations"]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        batch["image"] = image

        return batch

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values

        # prepare text input
        entities = []
        relations = []
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["labels"]
            del feature["original_image"]
            entities.append(feature["entities"])
            del feature["entities"]
            relations.append(feature["relations"])
            del feature["relations"]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        batch["image"] = image
        batch["entities"] = entities
        batch["relations"] = relations

        return batch



if __name__=='__main__':
    with open('generator_config.json') as f:
        configs = json.load(f)
    
    datasetGenerator = DatasetGenerator(**configs)
    datasetGenerator.generate_Dataset(tokenizer_dir='./tokenizer_added_tokens', model_dir='./model_added_tokens')
    #datasetGenerator.test_gnerator(tokenizer_path='trained_models/august_23/tokenizer_added_tokens', model_path = './trained_models/august_23/ts_finetuned', file_index = 2)
    datasetGenerator.test_gnerator(tokenizer_path='./tokenizer_added_tokens', model_path = './model_added_tokens', file_index = 23)
    #datasetGenerator.generate_Dataset(tokenizer_dir='./tokenizer_added_tokens', model_dir='./model_added_tokens')
    #datasetGenerator.test_gnerator(tokenizer_path='./tokenizer_added_tokens', model_path = './model_added_tokens', file_index = 1)
    
    configs = {key:val for key,val in configs.items() if key not in ("num_files", "clear_all_old_files", "clear_old_files_type", "datasets_init_configs")}
    custom_dataset = Custom_Dataset(**configs)
    #print(len(custom_dataset))
    
    model = LayoutLMv2ForRelationExtraction.from_pretrained('./model_added_tokens')
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_added_tokens')
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=True)
    
    data_collator = DataCollatorForTokenClassification(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=512,
    )
    

    #dataloader = DataLoader(custom_dataset, batch_size=1, collate_fn=data_collator)
    #for step, i in enumerate(dataloader):
    #    print(i)
    #    print('-------')
    #    break'''

    
            
                
            
                
            
            
            