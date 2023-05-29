import argparse

from transformers import AutoTokenizer, LayoutLMv2ForRelationExtraction
import numpy as np


#tokenizer = AutoTokenizer.from_pretrained('layout_xlm_base_tokenizer/')
#model = LayoutLMv2ForRelationExtraction.from_pretrained('layout_xlm_base_model/')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_temp_path', default='tokens_temp.npy')
    parser.add_argument('--bboxes_temp_path', default='bboxes_temp.npy')
    parser.add_argument('--base_model_path', default = 'layout_xlm_base_model')
    parser.add_argument('--base_tokenizer_path', default = 'layout_xlm_base_tokenizer')
    
    args = parser.parse_args()
    
    model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutxlm-base')
    
    tokens = np.load(args.tokens_temp_path)
    bboxes = np.load(args.bboxes_temp_path)
    
    new_tokens = tokens
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    
    
    
    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))
    tokenizer.save_pretrained(args.base_tokenizer_path) 
    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(args.base_model_path)
    
    new_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_path)
    input_ids = new_tokenizer.encode(text = tokens, boxes = bboxes, is_pretokenized=False)  
    print(f' Inside script lenght of input ids {len(input_ids)}')
    