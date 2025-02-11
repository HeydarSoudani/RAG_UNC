from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import TruthTorchLM as ttlm
import torch


class CorrectnessEvaluator(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, question_text:str, generated_text: str,  ground_truth_text: list[str], seed:int = None) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")

class ExactMatch(CorrectnessEvaluator):
    def __init__(self):
        pass
    
    def __call__(self, question_text:str, generated_text: str,  ground_truths: list[str], seed:int = None) -> bool:
        is_correct = 0
        for pa in ground_truths:
            generated_text = generated_text.strip()
            if pa in generated_text or pa.lower() in generated_text or pa.capitalize() in generated_text:
                is_correct = 1
                break
        
        return is_correct
    
    def __str__(self):
        return f"EM"
    


# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
api_model = "gpt-4o"

n_generations = 10
lars = ttlm.truth_methods.LARS(ue_type="semantic_entropy") # , number_of_generations=n_generations
sar = ttlm.truth_methods.SAR()
mars = ttlm.truth_methods.MARS() # number_of_generations=n_generations
se = ttlm.truth_methods.SemanticEntropy()
pe = ttlm.truth_methods.Entropy()
confidence = ttlm.truth_methods.Confidence() # number_of_generations=n_generations
self_detection = ttlm.truth_methods.SelfDetection(number_of_questions=5)
truth_methods = [lars, mars, sar, se, pe, confidence, self_detection]

# model_judge = ttlm.evaluators.ModelJudge('gpt-4o-mini')
# model_judge = ttlm.evaluators.ROUGE()
model_judge = ExactMatch()

results = ttlm.evaluate_truth_method(
    dataset='trivia_qa',
    model=model,
    truth_methods=truth_methods,
    eval_metrics=['auroc', 'accuracy', 'prr'],
    tokenizer=tokenizer,
    size_of_data=30,
    correctness_evaluator=model_judge,
    max_new_tokens=64
)
# print(results)
for i in range(len(results['eval_list'])):
    print(results['output_dict']['truth_methods'][i],results['eval_list'][i])


# python 2_truth_torch_framework/main.py
