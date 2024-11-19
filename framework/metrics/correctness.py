
import re
import torch
import evaluate
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import pipeline
from bert_score import BERTScorer
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


class RougeScore():
    def __init__(self):
        self.rouge = evaluate.load('rouge')
    
    def __call__(self, reference_answers, candidate):
        score = self.rouge.compute(
            predictions=[candidate],
            references=[reference_answers]
        )
        return score

class ExactMatch():
    def __init__(self):
        pass
    
    def __call__(self, reference_answers, candidate):
        is_correct = False
        for pa in reference_answers:
            if pa in candidate or pa.lower() in candidate or pa.capitalize() in candidate:
                is_correct = True
                break
        
        return is_correct

# ref: https://haticeozbolat17.medium.com/text-summarization-how-to-calculate-bertscore-771a51022964
class BertScore():
    def __init__(self):
        self.scorer = BERTScorer(model_type='bert-base-uncased')
    
    def __call__(self, reference_answers, candidate):
        bert_score = {'P': 0.0, 'R': 0.0, 'F1': 0.0}
        for answer in reference_answers:
            P, R, F1 = self.scorer.score([candidate], [answer])
            if float(F1[0]) > bert_score['F1']:
                bert_score = {'P': float(P[0]), 'R': float(R[0]), 'F1': float(F1[0])}
        
        return bert_score

class BemScore():
    def __init__(self):
        VOCAB_PATH = 'baselines/MARS/vocab.txt'  #@param {type:"string"}
        tf.config.set_visible_devices([], 'GPU')
        vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=VOCAB_PATH,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ), 
            num_oov_buckets=1)
        self.cls_id, self.sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
        self.bert_tokenizer = text.BertTokenizer(
            vocab_lookup_table=vocab_table, 
            token_out_type=tf.int64, 
            preserve_unused_token=True, 
            lower_case=True
        )
        # self.bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
        # self.bem = hub.load('https://www.kaggle.com/models/google/bert/TensorFlow2/answer-equivalence-bem/1')
        # self.bem = hub.load('https://www.kaggle.com/models/google/bert/TensorFlow2/answer-equivalence-bem/1')
        self.bem = hub.load('framework/metrics/bert-tensorflow2-answer-equivalence-bem-v1')    
    
    def bertify_example(self, example):
        question = self.bert_tokenizer.tokenize(example['question']).merge_dims(1, 2)
        reference = self.bert_tokenizer.tokenize(example['reference']).merge_dims(1, 2)
        candidate = self.bert_tokenizer.tokenize(example['candidate']).merge_dims(1, 2)

        input_ids, segment_ids = text.combine_segments(
            (candidate, reference, question), self.cls_id, self.sep_id)

        return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}

    def pad(self, a, length=512):
        return np.append(a, np.zeros(length - a.shape[-1], np.int32))
    
    def bertify_examples(self, examples):
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = self.bertify_example(example)
            input_ids.append(self.pad(example_inputs['input_ids']))
            segment_ids.append(self.pad(example_inputs['segment_ids']))

        return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}

    def __call__(self, question, reference_answers, candidate):
        
        bem_score = 0.0
        for answer in reference_answers:
            examples = [{
                'question': question,
                'reference': answer,
                'candidate': candidate
            }]
            inputs = self.bertify_examples(examples)
            raw_outputs = self.bem(inputs)
            # They can be transformed into a classification 'probability' like so:
            score = float(softmax(np.squeeze(raw_outputs))[1])
            bem_score = max(score, bem_score)
        
        return bem_score


class PromptFormatter:
    def __init__(self, model_id, sys_prompt=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        else:
            self.sys_prompt = "You are a helpful assistant."
        
    def format_prompt(self, prompt: str) -> list:
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class LLamaScore():
    def __init__(self, model_name, device, max_tokens=4, sys_prompt=None):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.max_new_tokens = max_tokens
        self.formatter = PromptFormatter(model_name, sys_prompt)
        self.max_retries = 5
        

    def prompt_generator(self, question, reference_answers, candidate):
        reference_answers_txt = ', '.join(reference_answers)
        prompt = "You will behave as a question-answer evaluator. I will give you a question, the ground truth of the question and a generated answer by a language model. You will output 'CORRECT' if the generated answer is correct regarding question and ground truth. Otherwise, output 'FALSE'.\n"
        prompt += f"Question: {question}\n"
        prompt += f"Ground Truth: {reference_answers_txt}\n"
        prompt += f"Generated Answer: {candidate}\n"
        prompt += 'Output: '
        return prompt

    def score_extractor(self, generated_text):
        # generated_text = generated_text.upper()
        match = re.search(r'\b(CORRECT|FALSE)\b', generated_text)
        if match:
            return match.group(0)  # Return the matched text ("CORRECT" or "FALSE")
        else:
            return None
    
    
    def __call__(self, question, reference_answers, candidate):
        is_correct = False
        
        prompt = self.prompt_generator(question, reference_answers, candidate)
        # print(prompt)
        input_ = self.tokenizer(prompt, padding=False, truncation=False)
        input_ids = torch.tensor(input_['input_ids']).to(self.device).reshape(1, -1)
    
        for i in range(self.max_retries):
            try:
                generation = self.model.generate(
                    input_ids,
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )
                outputs = self.tokenizer.decode(generation[0, len(input_ids[0]):], skip_special_tokens=True)
                score = self.score_extractor(outputs)
                
                if score == "CORRECT":
                    is_correct = True
                    return is_correct
                elif score == "FALSE":
                    is_correct = False
                    return is_correct
                else:
                    continue

            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{self.max_retries}")
                continue
        
        is_correct = None
        return is_correct
        
