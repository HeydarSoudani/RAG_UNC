
import json
import random
import functools
from tqdm import tqdm
from datasets import Dataset


def convert_jsonl_to_dataset_static_psg(dataset, split):
    dataset_name = dataset.split('_')[0]
    data_file = f"processed_datasets/{dataset_name}_{split}.jsonl"
    # data_file = f"processed_datasets/{dataset_name}_{split}_with_srag.jsonl"
    
    data = []
    with open(data_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Converting dataset ..."):
            item = json.loads(line.strip())
            
            if dataset in ['webquestions', 'trivia', 'nq', 'squad1', 'nqgold', 'popqa']:
                data.append({
                    "question_id": item["id"],
                    "question": item["question"],
                    "answers": item["answers"],
                    "positive_ctxs": item["positive_ctxs"],
                    "negative_ctxs": item["negative_ctxs"],
                    "hard_negative_ctxs": item["hard_negative_ctxs"],
                    
                    # "positive_ctxs_sentences": item["positive_ctxs_sentences"],
                    # "negative_ctxs_sentences": item["negative_ctxs_sentences"],
                    # "hard_negative_ctxs_sentences": item["hard_negative_ctxs_sentences"],
                })
            elif dataset in ['hotpotqa', '2wikimultihopqa', 'musique']:
                data.append({
                    "question_id": item["id"],
                    "question": item["question"],
                    "answers": item["answers"],
                    "positive_ctxs": item["positive_ctxs"],
                    "negative_ctxs": item["negative_ctxs"]
                })
            elif dataset in ['topicoqa_org', 'topicoqa_his', 'topicoqa_rw']:
                question = item["org_question"] if dataset=='topicoqa_org' else item["his_question"] if dataset=='topicoqa_his' else item["rw_question"] if dataset=='topicoqa_rw' else ''
                data.append({
                    "question_id": item["id"],
                    "question": question,
                    "answers": item["answers"],
                    "positive_ctxs": item["positive_ctxs"],
                    "negative_ctxs": item["negative_ctxs"]
                })

    dataset_obj = Dataset.from_list(data)
    return dataset_obj

def convert_jsonl_to_dataset_retrieved_psg(dataset, split, retriever_name):
    dataset_name = dataset.split('_')[0]
    dataset_file = f"processed_datasets/{dataset_name}_{split}.jsonl"
    retrieved_docs_file = f"processed_datasets/{dataset_name}_{split}_{retriever_name}.jsonl"
    
    retrieved_docs = {}
    with open(retrieved_docs_file, 'r') as file:
        for line in file:
            sample = json.loads(line)
            retrieved_docs[sample['id']] = sample["contexts"]
    
    data = []
    with open(dataset_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Converting dataset ..."):
            item = json.loads(line.strip())
            
            data.append({
                "question_id": item["id"],
                "question": item["question"],
                "answers": item["answers"],
                "ctxs": retrieved_docs[item['id']],
            })
    
    dataset_obj = Dataset.from_list(data)
    return dataset_obj

def load_instructions(prompt_format):
    instructions_file = 'processed_datasets/prompt_files/instructions.txt'
    with open(instructions_file, "r") as file:
        lines = file.readlines()
    
    if prompt_format == 'only_q':
        instruction = lines[0].strip()
    else:
        instruction = lines[1].strip()
        
    return instruction

def load_examples(dataset_name):
    examples_file = 'processed_datasets/prompt_files/examples.json'
    with open(examples_file, 'r') as f:
        examples = json.load(f)
    return examples[dataset_name]


@functools.lru_cache(1)
class RAGDataset:
    def __init__(self, tokenizer, prompt_format, dataset_name='webquestions', split='dev'):
        self.dataset_name = dataset_name
        self.prompt_format = prompt_format
        self.tokenizer = tokenizer
        
        if self.prompt_format in ['only_q', 'q_positive', 'q_negative']:
            self.dataset = convert_jsonl_to_dataset_static_psg(self.dataset_name, split)
        else:
            retriever_name = self.prompt_format.split('_')[0]
            self.dataset = convert_jsonl_to_dataset_retrieved_psg(self.dataset_name, split, retriever_name)
        
        self.max_tokens = 3900
        self.instruction = load_instructions(self.prompt_format)
        self.examples = load_examples(self.dataset_name)
        
    def generate_prompt_wo_ctx(self, query):
        prompt_text = f"""{self.instruction}\n\n"""
        for example in self.examples:
            prompt_text += f"Question: {example['query']}\n"
            prompt_text += f"Answer: {example['answer']}\n\n"
        prompt_text += f"Question: {query}\n"
        prompt_text += f"Answer: "
        
        return prompt_text
    
    def generate_prompt_with_ctx(self, query, context):
        prompt_text = f"""{self.instruction}\n\n"""
        for example in self.examples:
            prompt_text += f"Document:\n{example['context']}\n"
            prompt_text += f"Question: {example['query']}\n"
            prompt_text += f"Answer: {example['answer']}\n\n"
        prompt_text += f"Document:\n{context}\n"
        prompt_text += f"Question: {query}\n"
        prompt_text += f"Answer: "
        
        return prompt_text
    
    # def generate_prompt(self, question, positive_ctxs, negative_ctxs, hard_negative_ctxs):
    def generate_prompt(self, example):
        
        if self.prompt_format == 'only_q':
            similarity_score = 1.0
            prompt_text = self.generate_prompt_wo_ctx(example['question'])
            return prompt_text, similarity_score
        
        elif self.prompt_format == 'q_positive':
            if len(example['positive_ctxs']) > 0:
                # pos_passages = [psg['text'] for psg in positive_ctxs]
                # pos_passages_text = '\n'.join(pos_passages)
                # pos_passages_tokens = self.tokenizer(pos_passages_text, return_tensors='pt', truncation=False)['input_ids']
                # pos_passages_truncated_tokens = pos_passages_tokens[0][:self.max_tokens]
                # truncated_pos_passages_text = self.tokenizer.decode(pos_passages_truncated_tokens, skip_special_tokens=True)
                selected_context = random.choice(example['positive_ctxs'][:2])
                prompt_text = self.generate_prompt_with_ctx(example['question'], selected_context['text'])
                
                if 'similarity_score' in selected_context:
                    sim_score = selected_context['similarity_score']
                else:
                    sim_score = 1.0
                
                return prompt_text, sim_score
            
            else:
                similarity_score = 1.0
                prompt_text = self.generate_prompt_wo_ctx(example['question'])
                return prompt_text, similarity_score
        
        elif self.prompt_format == 'q_negative':
            if len(example['negative_ctxs']) > 0:
                selected_context = random.choice(example['negative_ctxs'])
                prompt_text = self.generate_prompt_with_ctx(example['question'], selected_context['text'])
                
                if 'similarity_score' in selected_context:
                    sim_score = selected_context['similarity_score']
                else:
                    sim_score = 1.0
                    
                return prompt_text, sim_score

            else:
                if len(example['hard_negative_ctxs']) > 0:
                    selected_context = random.choice(example['hard_negative_ctxs'])
                    prompt_text = self.generate_prompt_with_ctx(example['question'], selected_context['text'])
                    
                    if 'similarity_score' in selected_context:
                        sim_score = selected_context['similarity_score']
                    else:
                        sim_score = 1.0
                        
                    return prompt_text, sim_score
                    
                    
                else:
                    similarity_score = 1.0
                    prompt_text = self.generate_prompt_wo_ctx(example['question'])
                    return prompt_text, similarity_score
    
        else: # retrived dataset
            # selected_context = random.choice(example['ctxs'])
            selected_context = '\n'.join([ctx['context'] for ctx in example['ctxs'][:1]])
            sim_score = 1.0
            prompt_text = self.generate_prompt_with_ctx(example['question'], selected_context)
            
            return prompt_text, sim_score
        
        # else:
        #     raise ValueError(f"Prompt format {self.prompt_format} not supported")
    
    def get_dataset(self, add_prompt=None):
        
        def process_instance(example):
            # https://github.com/zlin7/UQ-NLG
            
            all_answers = example.pop('answers')
            example['answers'] = all_answers
            
            prompt, sim_score = self.generate_prompt(example)
            # prompt, sim_score = self.generate_prompt(
            #     example['question'],
            #     example['positive_ctxs'], example['negative_ctxs'], example['hard_negative_ctxs']
            # )
            example['prompt'] = prompt
            inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
            outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            
            example['input_ids'] = inputs['input_ids']
            example["attention_mask"] = inputs.attention_mask
            example["labels"] = outputs.input_ids.copy()
            example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
            example['similarity_score'] = sim_score
            
            return example

        def process_instance_v2(example):
            all_answers = example.pop('answers')
            example['answers'] = all_answers
            
            prompts, sim_score = self.generate_prompt(
                example['question'],
                example['positive_ctxs'], example['negative_ctxs'], example['hard_negative_ctxs'],
                example['positive_ctxs_sentences'], example['negative_ctxs_sentences'], example['hard_negative_ctxs_sentences']
            )
            inputs_list = [self.tokenizer(prompt, padding=False, truncation=False) for prompt in prompts] 
            outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            
            example['input_ids'] = [inputs['input_ids'] for inputs in inputs_list]
            example["attention_mask"] = [inputs.attention_mask for inputs in inputs_list]
            example["labels"] = outputs.input_ids.copy()
            example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
            example['prompts'] = prompts
            example['similarity_score'] = sim_score
            
            return example

        self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
        # self.dataset = self.dataset.map(process_instance_v2, load_from_cache_file=False)
        
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True)
        
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]















# @functools.lru_cache(1)
# class TriviaQA:
#     def __init__(self, tokenizer, prompt_format, split='dev'):
#         self.dataset_name = 'trivia'
#         self.dataset = convert_jsonl_to_dataset(self.dataset_name, split)
#         self.tokenizer = tokenizer
#         self.prompt_format = prompt_format
#         self.max_tokens = 3900
    
#     def generate_prompt(self, question, positive_ctxs, negative_ctxs, hard_negative_ctxs):
#         if self.prompt_format == 'only_q':
#             return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
        
#         elif self.prompt_format == 'q_positive':
#             if len(positive_ctxs) > 0:
#                 pos_passages = [psg['text'] for psg in positive_ctxs]
#                 pos_passages_text = '\n'.join(pos_passages)
#                 pos_passages_tokens = self.tokenizer(pos_passages_text, return_tensors='pt', truncation=False)['input_ids']
#                 pos_passages_truncated_tokens = pos_passages_tokens[0][:self.max_tokens]
#                 truncated_pos_passages_text = self.tokenizer.decode(pos_passages_truncated_tokens, skip_special_tokens=True)
                
#                 return """You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                     Documents: {}
#                     Question: {}
#                     Answer: """.format(truncated_pos_passages_text, question).replace('   ', '')
            
#             else:
#                 return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
            
            
#         elif self.prompt_format == 'q_negative':
#             if len(negative_ctxs) > 0:
#                 return f"""You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                     Documents:\n{random.choice(negative_ctxs)['text']}
#                     Question: {question}
#                     Answer: """.replace('   ', '')

#             else:
#                 if len(hard_negative_ctxs) > 0:
#                     # You are given a question and you MUST respond by EXTRACTING the answer (max 10 tokens) from the provided document.
#                     return f"""You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                                 Documents:\n{random.choice(hard_negative_ctxs)['text']}
#                                 Question: {question}
#                                 Answer: """.replace('   ', '')
#                 else:
#                     return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
    
#         else:
#             raise ValueError(f"Prompt format {self.prompt_format} not supported")
    
#     def get_dataset(self, add_prompt=None):
        
#         def process_instance(example):
#             # https://github.com/zlin7/UQ-NLG
            
#             all_answers = example.pop('answers')
#             example['answers'] = all_answers
            
#             prompt = self.generate_prompt(example['question'], example['positive_ctxs'], example['negative_ctxs'], example['hard_negative_ctxs'])
#             example['prompt'] = prompt
            
#             inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
#             outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            
#             example['input_ids'] = inputs['input_ids']
#             example["attention_mask"] = inputs.attention_mask
#             example["labels"] = outputs.input_ids.copy()
#             example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
#             return example

#         self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
#         self.dataset.set_format(
#             type="torch",
#             columns=["input_ids", "attention_mask", "labels"],
#             output_all_columns=True)
        
#         return self.dataset
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         return self.dataset[idx]

# @functools.lru_cache(1)
# class NaturalQuestions:
#     def __init__(self, tokenizer, prompt_format, split='dev'):
#         self.dataset_name = 'nq'
#         self.dataset = convert_jsonl_to_dataset(self.dataset_name, split)
#         self.tokenizer = tokenizer
#         self.prompt_format = prompt_format
#         self.max_tokens = 3900
    
#     def generate_prompt(self, question, positive_ctxs, negative_ctxs, hard_negative_ctxs):
#         if self.prompt_format == 'only_q':
#             return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
        
#         elif self.prompt_format == 'q_positive':
#             if len(positive_ctxs) > 0:
#                 pos_passages = [psg['text'] for psg in positive_ctxs]
#                 pos_passages_text = '\n'.join(pos_passages)
#                 pos_passages_tokens = self.tokenizer(pos_passages_text, return_tensors='pt', truncation=False)['input_ids']
#                 pos_passages_truncated_tokens = pos_passages_tokens[0][:self.max_tokens]
#                 truncated_pos_passages_text = self.tokenizer.decode(pos_passages_truncated_tokens, skip_special_tokens=True)
                
#                 return """You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                     Documents: {}
#                     Question: {}
#                     Answer: """.format(truncated_pos_passages_text, question).replace('   ', '')
            
#             else:
#                 return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
            
            
#         elif self.prompt_format == 'q_negative':
#             if len(negative_ctxs) > 0:
#                 return f"""You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                     Documents:\n{random.choice(negative_ctxs)['text']}
#                     Question: {question}
#                     Answer: """.replace('   ', '')

#             else:
#                 if len(hard_negative_ctxs) > 0:
#                     return f"""You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.
#                                 Documents:\n{random.choice(hard_negative_ctxs)['text']}
#                                 Question: {question}
#                                 Answer: """.replace('   ', '')
#                 else:
#                     return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
    
#         else:
#             raise ValueError(f"Prompt format {self.prompt_format} not supported")
    
#     def get_dataset(self, add_prompt=None):
        
#         def process_instance(example):
#             # https://github.com/zlin7/UQ-NLG
            
#             all_answers = example.pop('answers')
#             example['answers'] = all_answers
            
#             prompt = self.generate_prompt(example['question'], example['positive_ctxs'], example['negative_ctxs'], example['hard_negative_ctxs'])
#             example['prompt'] = prompt
            
#             inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
#             outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            
#             example['input_ids'] = inputs['input_ids']
#             example["attention_mask"] = inputs.attention_mask
#             example["labels"] = outputs.input_ids.copy()
#             example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
#             return example

#         self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
#         self.dataset.set_format(
#             type="torch",
#             columns=["input_ids", "attention_mask", "labels"],
#             output_all_columns=True)
        
#         return self.dataset
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         return self.dataset[idx]

# @functools.lru_cache(1)
# class Squad:
    
#     def __init__(self, tokenizer, prompt_format, split='dev'):
#         self.dataset_name = 'nq'
#         self.dataset = convert_jsonl_to_dataset(self.dataset_name, split)
#         self.tokenizer = tokenizer
#         self.prompt_format = prompt_format
#         self.max_tokens = 3900
    
#     def generate_prompt(self, question, positive_ctxs, negative_ctxs, hard_negative_ctxs):
#         if self.prompt_format == 'only_q':
#             return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
        
#         elif self.prompt_format == 'q_positive':
#             if len(positive_ctxs) > 0:
#                 pos_passages = [psg['text'] for psg in positive_ctxs]
#                 pos_passages_text = '\n'.join(pos_passages)
#                 pos_passages_tokens = self.tokenizer(pos_passages_text, return_tensors='pt', truncation=False)['input_ids']
#                 pos_passages_truncated_tokens = pos_passages_tokens[0][:self.max_tokens]
#                 truncated_pos_passages_text = self.tokenizer.decode(pos_passages_truncated_tokens, skip_special_tokens=True)
                
#                 return """You are given a question and you MUST respond by EXTRACTING the answer (max 10 tokens) from the provided document.
#                     Documents: {}
#                     Question: {}
#                     Answer: """.format(truncated_pos_passages_text, question).replace('   ', '')
            
#             else:
#                 return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
            
            
#         elif self.prompt_format == 'q_negative':
#             if len(negative_ctxs) > 0:
#                 return f"""You are given a question and you MUST respond by EXTRACTING the answer (max 10 tokens) from the provided document.
#                     Documents:\n{random.choice(negative_ctxs)['text']}
#                     Question: {question}
#                     Answer: """.replace('   ', '')

#             else:
#                 if len(hard_negative_ctxs) > 0:
#                     return f"""You are given a question and you MUST respond by EXTRACTING the answer (max 10 tokens) from the provided document.
#                                 Documents:\n{random.choice(hard_negative_ctxs)['text']}
#                                 Question: {question}
#                                 Answer: """.replace('   ', '')
#                 else:
#                     return f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
    
#         else:
#             raise ValueError(f"Prompt format {self.prompt_format} not supported")
    
#     def get_dataset(self, add_prompt=None):
        
#         def process_instance(example):
#             # https://github.com/zlin7/UQ-NLG
            
#             all_answers = example.pop('answers')
#             example['answers'] = all_answers
            
#             prompt = self.generate_prompt(example['question'], example['positive_ctxs'], example['negative_ctxs'], example['hard_negative_ctxs'])
#             example['prompt'] = prompt
            
#             inputs = self.tokenizer(example['prompt'], padding=False, truncation=False)
#             outputs = self.tokenizer(all_answers[0], padding=False, truncation=False)
            
#             example['input_ids'] = inputs['input_ids']
#             example["attention_mask"] = inputs.attention_mask
#             example["labels"] = outputs.input_ids.copy()
#             example["labels"] = [-100 if _ == self.tokenizer.pad_token_id else _ for _ in example["labels"]]
#             return example

#         self.dataset = self.dataset.map(process_instance, load_from_cache_file=False)
#         self.dataset.set_format(
#             type="torch",
#             columns=["input_ids", "attention_mask", "labels"],
#             output_all_columns=True)
        
#         return self.dataset
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         return self.dataset[idx]

