
# Ref: https://github.com/facebookresearch/contriever/blob/main/src/evaluation.py


import collections
import logging
import regex
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""
QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for i, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


