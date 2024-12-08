#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils.clustering as pc
from utils.utils import set_seed

def get_uncertainty_bb(args):
    print("\n--- Phase 2: Get BB Uncertainty ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    
    # === Define IN/OUT files ========================
    model = args.model.split('/')[-1]
    base_dir = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}'
    generation_file = f'{base_dir}/{model}_{args.temperature}_cleaned_generation.pkl'
    uncertainty_output_file = f'{base_dir}/{model}_{args.temperature}_uncertainty_bb_generation.pkl'
    uncertainty_output_jsonl_file = f'{base_dir}/components_output/{model}_{args.temperature}_uncertainty_bb_generation.jsonl'
    os.makedirs(f'{base_dir}/components_output', exist_ok=True)

    
    with open(generation_file, 'rb') as infile:
        sequences = pickle.load(infile)
    
    # === Functions/Classes ==========================
    class NLIModel:
    
        def __init__(self, model_name='microsoft/deberta-large-mnli', **kwargs):
            self.pipe = pipeline(model=model_name, **kwargs)
        
        @torch.no_grad()
        def classify(self, prompt, responses, **kwargs):
            '''
            Input:
                prompt: a string-formatted prompt p
                responses: responses [r_1, ..., r_n]
            Output:
                a dictionary contains a mapping of response indexing and a n*n similarity matrix
            '''
            # https://github.com/lorenzkuhn/semantic_uncertainty
            # https://github.com/zlin7/UQ-NLG
            semantic_set_ids = {resp: i for i, resp in enumerate(responses)}
            _rev_mapping = semantic_set_ids.copy()
            sim_mat_batch = torch.zeros((len(responses), len(responses),3))
            make_input = lambda x: dict(text=x[0],text_pair=x[1])
            for i, response_i in enumerate(responses):
                for j, response_j in enumerate(responses):
                    # if i == j: continue # may not needed
                    scores = self.pipe(make_input([f"{prompt} {response_i}", f"{prompt} {response_j}"]), return_all_scores=True, **kwargs)
                    sim_mat_batch[i,j] = torch.tensor([score['score'] for score in scores])
            return dict(
                mapping = [_rev_mapping[_] for _ in responses],
                sim_mat = sim_mat_batch
            )
        
        @torch.no_grad()
        def compare(self, prompt, response_1, response_2, **kwargs):
            prompt1 = dict(text=f'{prompt} {response_1}', text_pair=f'{prompt} {response_2}')
            prompt2 = dict(text=f'{prompt} {response_2}', text_pair=f'{prompt} {response_1}')
            logits_list = self.pipe([prompt1, prompt2], return_all_scores=True, **kwargs)
            logits = torch.tensor([[logit['score'] for logit in logits] for logits in logits_list])
            pred = 0 if logits.argmax(dim=1).min() == 0 else 1
            return {
                'deberta_prediction': pred,
                'prob': logits.cpu(),
                'pred': logits.cpu()
            }
    
    # TODO: replace CLIP with NLI
    # Ref: https://github.com/AoShuang92/css_uq_llms/blob/main/UQ-CSS/models/_load_model.py
    class CLIPModel_Text(nn.Module):
        def __init__(self, device):
            super(CLIPModel_Text, self).__init__()
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.config = model.config
            self.text_model = model.text_model
            self.text_projection = model.text_projection
            self.logit_scale = model.logit_scale
            self.device= device

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            return_loss = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            text_embeds = text_outputs[1]
            text_embeds = self.text_projection(text_embeds)

            # normalized features
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # print('text_embeds',text_embeds.shape)

            # cosine similarity as logits
            # prob_per_pair1 = text_embeds[0] * text_embeds[1] #torch.mm(text_embeds[0], text_embeds[1]) #* logit_scale\
            # prob_per_pair2 = text_embeds[2] * text_embeds[3]

            all_prob_pairs = []
            for i in range(int(text_embeds.shape[0]/2)):
                i = 2*i
                prob_per_pair_ = text_embeds[i] * text_embeds[i+1]
                all_prob_pairs.append(prob_per_pair_)

            return torch.stack(all_prob_pairs, dim = 0)

    class BlackBox():
        def __init__(self):
            return NotImplementedError

        def compute_scores(self):
            return NotImplementedError

    def spectral_projected(affinity_mode, batch_sim_mat, threshold=0.1):
        # sim_mats: list of similarity matrices using semantic similarity model or jacard similarity
        clusterer = pc.SpetralClustering(affinity_mode=affinity_mode, cluster=False, eigv_threshold=threshold)
        return [clusterer.proj(sim_mat) for sim_mat in batch_sim_mat]

    def jaccard_similarity(batch_sequences):
        '''
        Input:
            batch_sequences: a batch of sequences [[s_1^1, ..., s_{n_1}^1], ..., [s_1^1, ..., s_{n_B}^B]]
        Output:
            batch_sim_mat: a batch of real-valued similairty matrices [S^1, ..., S^B]
        '''
        batch_sim_mats = []
        for sequences in batch_sequences:
            wordy_sets = [set(seq.lower().split()) for seq in sequences]
            mat = np.eye(len(wordy_sets))
            for i, set_i in enumerate(wordy_sets):
                for j, set_j in enumerate(wordy_sets[i+1:], i+1):
                    mat[i,j] = mat[j,i] = len(set_i.intersection(set_j)) / max(len(set_i.union(set_j)),1)
            batch_sim_mats.append(mat)
        return batch_sim_mats

    class SemanticConsistency(BlackBox):
        def __init__(self, similarity_model=None, device='cuda'):
            self.device = device if device is not None else torch.device('cpu')
            if not similarity_model:
                self.similarity_model = NLIModel(device=device)
            else:
                self.similarity_model = similarity_model
        
        def similarity_mat(self, prompts, sequences):
            '''
            Input:
                prompts: a batch of prompt [p^1, ..., p^B]
                sequences: a batch of sequences [[s_1^1, ..., s_{n_1}^1], ..., [s_1^1, ..., s_{n_B}^B]]
            Output:
                batch_sim_mat: a batch of real-valued similairty matrices [S^1, ..., S^B]
            '''
            sims = [self.similarity_model.classify(prompts, seq) for seq in sequences]
            return [s['sim_mat'] for s in sims]

    class Eccentricity(BlackBox):
        def __init__(self, affinity_mode='disagreement', semantic_model=None, device='cuda:0'):
            self.affinity_mode = affinity_mode
            if affinity_mode != 'jaccard' and not semantic_model:
                self.sm = SemanticConsistency(NLIModel(device=device))
            elif affinity_mode != 'jaccard':
                self.sm = semantic_model
        
        def compute_scores(self, batch_prompts, batch_responses, batch_sim_mats=None, **kwargs):
            '''
            Input:
                batch_prompts: a batch of prompts[prompt_1, ..., prompt_B]
                batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
            Output:
                batch_U: a batch of uncertainties [U^1, ..., U^B]
                batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
            '''
            if batch_sim_mats is None:
                batch_sim_mats = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
            batch_projected = spectral_projected(self.affinity_mode, batch_sim_mats, threshold=0.1)
            batch_Cs = [-np.linalg.norm(projected-projected.mean(0)[None, :],2,axis=1) for projected in batch_projected]
            batch_U = [np.linalg.norm(projected-projected.mean(0)[None, :], 2) for projected in batch_projected]
            return batch_U, batch_Cs, batch_sim_mats
    
    class Degree(BlackBox):
        def __init__(self, affinity_mode='disagreement', semantic_model=None, device='cuda:0'):
            self.affinity_mode = affinity_mode
            if affinity_mode != 'jaccard' and not semantic_model:
                self.sm = SemanticConsistency(NLIModel(device=device))
        
        def compute_scores(self, batch_prompts, batch_responses, batch_sim_mats=None, **kwargs):
            '''
            Input:
                batch_prompts: a batch of prompts [p^1, ..., p^B]
                batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
            Output:
                batch_U: a batch of uncertainties [U^1, ..., U^B]
                batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
            '''
            if batch_sim_mats is None:
                batch_sim_mats = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
            batch_W = [pc.get_affinity_mat(sim_mat, self.affinity_mode) for sim_mat in batch_sim_mats]
            batch_Cs = [np.mean(W, axis=1) for W in batch_W]
            batch_U = [1/W.shape[0]-np.sum(W)/W.shape[0]**2 for W in batch_W]
            # print(batch_W)
            # print(batch_Cs)
            # print(batch_U)
            return batch_U, batch_Cs, batch_sim_mats

    class SpectralEigv(BlackBox):
        def __init__(self, affinity_mode='disagreement', temperature=1.0, semantic_model=None, adjust=False, device='cuda:0'):
            self.affinity_mode = affinity_mode
            self.temperature = temperature
            self.adjust = adjust
            if affinity_mode == 'jaccard':
                self.consistency = jaccard_similarity
            else:
                nlimodel = NLIModel(device=device)
                self.sm = SemanticConsistency(nlimodel)

        def compute_scores(self, batch_prompts, batch_responses, sim_mats=None, **kwargs):
            if sim_mats is None:
                sim_mats = jaccard_similarity(batch_responses) if self.affinity_mode == 'jaccard' else self.sm.similarity_mat(batch_prompts, batch_responses)
            clusterer = pc.SpetralClustering(affinity_mode=self.affinity_mode, eigv_threshold=None,
                                                    cluster=False, temperature=self.temperature)
            return [clusterer.get_eigvs(_).clip(0 if self.adjust else -1).sum() for _ in sim_mats], None, sim_mats

    # === Main loop ==================================
    ECC = Eccentricity(affinity_mode=args.affinity_mode, device=args.device)
    DEGREE = Degree(affinity_mode=args.affinity_mode, device=args.device, semantic_model=ECC.sm)
    SPECTRAL = SpectralEigv(affinity_mode=args.affinity_mode, device=args.device, semantic_model=ECC.sm)

    bb_unc_sequences = []
    with open(uncertainty_output_jsonl_file, 'w') as jl_ofile:
        for idx, sample in tqdm(enumerate(sequences)):
            
            # if idx == 1:
            #     break
            
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            generations = sample['cleaned_generated_texts']
            # print(question_id)
            unc_seq_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers,
                'generations': generations,
            }
            
            ecc_u, ecc_c, sim_mats = ECC.compute_scores([""], [generations])
            degree_u, degree_c, sim_mats = DEGREE.compute_scores([""], [generations], batch_sim_mats=sim_mats)
            spectral_u, spectral_c, sim_mats = SPECTRAL.compute_scores([""], [generations], sim_mats=sim_mats)
            
            # print(sim_mats)
            # print(degree_u)
            
            unc_seq_dict['degree_u'] = np.float64(degree_u[0])
            unc_seq_dict['ecc_u'] = np.float64(ecc_u[0])
            unc_seq_dict['spectral_u'] = np.float64(spectral_u[0])

            bb_unc_sequences.append(unc_seq_dict)
            
            # === Write in a jsonl file 
            bb_unc_sequence_jsl = {
                'id': question_id,
                'degree_u': unc_seq_dict['degree_u'],
                'ecc_u': unc_seq_dict['ecc_u'],
                'spectral_u': unc_seq_dict['spectral_u'],
                'question': question,
                'answers': reference_answers,
                'generated_texts': generations,
            }
            jl_ofile.write(json.dumps(bb_unc_sequence_jsl) + '\n')

    ### === Save the correctness result ============
    with open(uncertainty_output_file, 'wb') as ofile:
        pickle.dump(bb_unc_sequences, ofile)
    print(f"Results saved to {uncertainty_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="bem_score", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument("--output_file_postfix", type=str, default="")
    
    parser.add_argument('--num_generations_per_prompt', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
    parser.add_argument('--run_id', type=str, default='run_0')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    args.output_dir = "framework/run_output"
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    if args.main_prompt_format != 'only_q':
        args.second_prompt_format == 'only_q'
    
    set_seed(args.seed)
    get_uncertainty_bb(args)
    
    # python framework/run/get_uncertainty_blackbox.py