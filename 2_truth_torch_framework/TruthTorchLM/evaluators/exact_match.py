from .correctness_evaluator import CorrectnessEvaluator

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