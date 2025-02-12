import json
import random
from tqdm import tqdm

BASE_DIR = '/home/hsoudani/RAG_UNC'

def get_dataset(prompt_format, dataset_name, split, fraction_of_data_to_use):
    
    dataset_file = f"{BASE_DIR}/datasets/processed_files/{dataset_name}_{split}.jsonl"
    if prompt_format not in ['only_q', 'q_positive', 'q_negative']:
        retriever_name = prompt_format.split('_')[0]
        retrieved_docs_file = f"{BASE_DIR}/datasets/processed_files/{dataset_name}_{split}_{retriever_name}.jsonl"
        
        retrieved_docs = {}
        with open(retrieved_docs_file, 'r') as file:
            for line in file:
                sample = json.loads(line)
                retrieved_docs[sample['id']] = sample["contexts"]


    data = []
    with open(dataset_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Converting dataset ..."):
            item = json.loads(line.strip())
            
            # === Get context ===============================
            if prompt_format == 'only_q':
                context = ""
            elif prompt_format == 'q_positive':
                context = random.choice(item['positive_ctxs'][:2])['text']
            elif prompt_format == 'q_negative':
                if len(item['negative_ctxs']) > 0:
                    context = random.choice(item['negative_ctxs'])['text']
                else:
                    if len(item['hard_negative_ctxs']) > 0:
                        context = random.choice(item['hard_negative_ctxs'])['text']
                    else:
                        context = ""
            else:
                ctxs = retrieved_docs[item['id']]
                context = '\n'.join([ctx['context'] for ctx in ctxs[:1]])
            
            
            # === 
            data.append({
                "qid": item["id"],
                "question": item["question"],
                "ground_truths": item["answers"],
                "context": context
            })
    
        
    if fraction_of_data_to_use < 1.0:
        random.shuffle(data)
        subset_length = int(len(data) * fraction_of_data_to_use)
        test_data = data[:subset_length]
    else:
        test_data = data

    return test_data

