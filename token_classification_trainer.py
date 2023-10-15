from dataset_creator import *
from transformers import LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, LayoutLMv2ForTokenClassification
from datasets import load_metric
import numpy as np
from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForTokenClassification
from unilm.layoutlmft.layoutlmft.evaluation import re_score
from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer
import argparse

#logging.set_verbosity_error()

metric = load_metric("seqeval")
return_entity_level_metrics = True

labels = ['O', 'B-QUESTION', 'B-ANSWER', 'B-HEADER', 'I-ANSWER', 'I-QUESTION', 'I-HEADER']
id2label = {k:v for k,v in enumerate(labels)}
label2id = {v:k for k,v in enumerate(labels)}

def create_dataset(type:str = 'train', config_file = 'generator_config.json', root_dir:str=None):
    assert type in ('train', 'validation')
    with open(config_file) as f:
        configs = json.load(f)
        
    configs = {key:val for key,val in configs.items() if key not in ("num_files", "clear_all_old_files", "clear_old_files_type", "datasets_init_configs")}
    configs['type'] = type
    configs['root_dir'] = root_dir
    dataset = Custom_Dataset(**configs)
    
    return dataset

def unnormalize_box(bbox, width, height):
     #x1,y1,x2,y2 = yolobbox2bbox(bbox[0], bbox[1], bbox[2], bbox[3])
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
     
def compute_metrics_relation_extraction(p):
    pred_relations, gt_relations = p
    score = re_score(pred_relations, gt_relations, mode="boundaries")
    return score
     
def compute_metrics_token_classification(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    
def train_token_classification_model(tokenizer_path:str = None, model_path:str = None, batch_size :int = None, steps:int=None, model_output_dir:str=None, root_dir:str=None):
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutXLMTokenizer.from_pretrained(tokenizer_path, padding=True, truncation=True)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path,
                                                         id2label=id2label,
                                                         label2id=label2id, num_labels = 33)
    
    data_collator = DataCollatorForTokenClassification(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=512
    )
    
    train_dataset_custom = create_dataset(type='train', root_dir=root_dir)
    test_dataset_custom = create_dataset(type='validation', root_dir=root_dir)
    #print(train_dataset_custom.keys())
    

    args = TrainingArguments(
        output_dir=model_output_dir, # name of directory to store the checkpoints
        overwrite_output_dir=True,
        max_steps=steps, # we train for a maximum of 1,000 batches
        warmup_ratio=0.1, # we warmup a bit
        disable_tqdm=False,
        # fp16=True, # we use mixed precision (less memory consumption)
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size // 2,
        #per_device_train_batch_size=2,
        #per_device_eval_batch_size=2,
        learning_rate=1e-5,
        remove_unused_columns=False,
        push_to_hub=False, # we'd like to push our model to the hub during training 
        save_strategy="no"
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset_custom,
        eval_dataset=test_dataset_custom,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_token_classification,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    
def train_realation_extracion_model(tokenizer_path:str = None, model_path:str = None, batch_size :int = None, steps:int=None, model_output_dir:str=None, root_dir:str=None):
    tokenizer = LayoutXLMTokenizer.from_pretrained(tokenizer_path)
    model = LayoutLMv2ForRelationExtraction.from_pretrained(model_path)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    
    data_collator = DataCollatorForKeyValueExtraction(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )
    
    train_dataset= create_dataset(type='train', root_dir=root_dir)
    test_dataset= create_dataset(type='validation', root_dir=root_dir)
    
    training_args = TrainingArguments(output_dir=model_output_dir,
                                  overwrite_output_dir=True,
                                  remove_unused_columns=False,
                                  # fp16=True, -> led to a loss of 0
                                  max_steps=steps,
                                  per_device_train_batch_size = batch_size,
                                  per_device_eval_batch_size = batch_size //2 ,
                                  dataloader_num_workers=4,
                                  warmup_ratio=0.1,
                                  learning_rate=1e-5,
                                  push_to_hub=False,
                                  no_cuda=False,
                                  save_strategy="no",
                                )

    trainer = XfunReTrainer(
    #trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_relation_extraction,
    )
   #results = trainer.train()
    trainer.train()
    #print_summary(results)
    trainer.save_model(model_output_dir)


def model_eval_token_classification(model_input_dir:str=None, train_dataset:Custom_Dataset=None, test_dataset:Custom_Dataset=None, tokenizer_path:str=None, id2label:dict=None, label2id:dict=None):
    #print(os.listdir(model_input_dir))
    sub_dirs_model_input =os.listdir(model_input_dir)
    flag = True if len([sub_dir for sub_dir in sub_dirs_model_input if 'checkpoint' in sub_dir]) > 0 else False
    if flag:
        sub_dirs = [sub_dir for sub_dir in sub_dirs_model_input if 'checkpoint' in sub_dir]
        latest_checkpoint = sub_dirs[-1]
        model_input_dir = os.path.join(model_input_dir, latest_checkpoint)
        
    model = AutoModelForTokenClassification.from_pretrained(model_input_dir)
    tokenizer = LayoutXLMTokenizer.from_pretrained(tokenizer_path)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    
    data_collator = DataCollatorForTokenClassification(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )
    
    args = TrainingArguments(
        output_dir='dummy_eval', # name of directory to store the checkpoints
        overwrite_output_dir=True,
        #max_steps=steps, # we train for a maximum of 1,000 batches
        warmup_ratio=0.1, # we warmup a bit
        # fp16=True, # we use mixed precision (less memory consumption)
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        remove_unused_columns=False,
        push_to_hub=False, # we'd like to push our model to the hub during training 
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_token_classification,
    )
    
    predictions, labels, metrics = trainer.predict(test_dataset)
    
    return predictions, labels, metrics

def model_eval_relation_extraction(model_input_dir:str=None, train_dataset:Custom_Dataset=None, test_dataset:Custom_Dataset=None, tokenizer_path:str=None, batch_size:int = 4):
    #print(os.listdir(model_input_dir))
    sub_dirs_model_input =os.listdir(model_input_dir)
    flag = True if len([sub_dir for sub_dir in sub_dirs_model_input if 'checkpoint' in sub_dir]) > 0 else False
    if flag:
        sub_dirs = [sub_dir for sub_dir in sub_dirs_model_input if 'checkpoint' in sub_dir]
        latest_checkpoint = sub_dirs[-1]
        print(f' latest checkpoint {latest_checkpoint}')
        model_input_dir = os.path.join(model_input_dir, latest_checkpoint)
        
    model = LayoutLMv2ForRelationExtraction.from_pretrained(model_input_dir)
    tokenizer = LayoutXLMTokenizer.from_pretrained(tokenizer_path)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    
    data_collator = DataCollatorForKeyValueExtraction(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=512,
    )
    
    args = TrainingArguments(
        output_dir='dummy_eval', # name of directory to store the checkpoints
        overwrite_output_dir=True,
        #max_steps=steps, # we train for a maximum of 1,000 batches
        warmup_ratio=0.1, # we warmup a bit
        # fp16=True, # we use mixed precision (less memory consumption)
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-5,
        remove_unused_columns=False,
        push_to_hub=False, # we'd like to push our model to the hub during training 
    )

    # Initialize our Trainer
    trainer = XfunReTrainer(
    #trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_relation_extraction,
    )
    
    #predictions, labels, metrics = trainer.predict(test_dataset)
    results = trainer.evaluate()
    return results

def change_labels():
    global id2label
    global label2id
    with open('id2label_no_re.json', 'r') as f:
        id2label = json.load(f)
        id2label = {int(key):val for key,val in id2label.items()}
    with open('label2id_no_re.json', 'r') as f:
        label2id = json.load(f)
        label2id = {key:int(val) for key,val in label2id.items()}

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog='TS_RE_trainer',
                    description='trains token classification or relation extraction model')
    parser.add_argument('--mode', type=str, default='train', required=True)
    parser.add_argument('--type', type=str, required=True, default='ts')
    parser.add_argument('--batch_size', type=int, required=True, default=5)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--tokenizer_path', type=str, required = True)
    parser.add_argument('--model_input_dir', type=str, required=True)
    parser.add_argument('--finetune_dir', type=str, required=True)
    parser.add_argument('--ts_type', type=str, default='re')
    parser.add_argument('--dataset_root_dir', type=str, required=True)
    
    args = parser.parse_args()
    assert args.mode in ('train', 'eval')
    assert args.type in ['ts', 're']
    assert args.ts_type in ('re', 'no_re')

    if args.mode =='train':
        if args.type == 'ts':
            if args.ts_type == 'no_re':
                change_labels()
            
            train_token_classification_model(tokenizer_path=args.tokenizer_path, model_path=args.model_input_dir, 
                batch_size=args.batch_size, steps=args.steps, model_output_dir=args.finetune_dir, root_dir=args.dataset_root_dir)
        else:
            train_realation_extracion_model(tokenizer_path=args.tokenizer_path, model_path=args.model_input_dir, 
                batch_size=args.batch_size, steps=args.steps, model_output_dir=args.finetune_dir, root_dir=args.dataset_root_dir)
    else:
        train_dataset = create_dataset(type='train', root_dir=args.dataset_root_dir)
        test_dataset = create_dataset(type='validation', root_dir=args.dataset_root_dir)
        if args.type == 'ts':
            if args.ts_type == 'no_re':
                change_labels()
            _, _, metrics = model_eval_token_classification(model_input_dir=args.model_input_dir, train_dataset=train_dataset, test_dataset= train_dataset, tokenizer_path=args.tokenizer_path)
            print(metrics)
        else:
            results = model_eval_relation_extraction(model_input_dir=args.model_input_dir, train_dataset=train_dataset, test_dataset= test_dataset, tokenizer_path=args.tokenizer_path)
            print(results)
    #train_dataset = create_dataset(type='train')
    #test_dataset = create_dataset(type='validation')
    #print(train_dataset[0].keys())
    #print(test_dataset[0].keys())
    #_, _, metrics = model_eval(model_input_dir='token_classification_finetuned', train_dataset=train_dataset, test_dataset= test_dataset, tokenizer_path='tokenizer_added_tokens')
    #print(metrics)
