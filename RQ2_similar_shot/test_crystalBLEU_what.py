


import numpy as np
from collections import Counter
from nltk.util import ngrams
# 1. Import CrystalBLEU
from crystalbleu import corpus_bleu
import jsonlines

import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from joblib import Parallel, delayed
from functools import lru_cache
from cachetools import cached, LRUCache, TTLCache
from cachetools.keys import hashkey
import multiprocessing

from joblib import Memory
location = '~/.cache/crystal_bleu'
memory = Memory(location, verbose=0, compress=True)

@memory.cache
def get_database(file_path_how):
    print('start--------read-----data')
    datas = []
    with jsonlines.open(file_path_how, 'r')as f:
        for dat in f:
            datas.append(dat)
    
    return datas


def clean_str(s: str) -> str:
    s = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', s)
    s = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def tokenize(datas):
    codes = []
    for dat in datas:
        code_str = dat['code']
        code_str = clean_str(code_str)
        codes.append(code_str)
    text = " ".join(codes)
    # text = " ".join(datas)
    # 假设你有一段文本
    # text = "This is a sample sentence."
    print('start------tokenized_corpus')
    # 对文本进行分词
    tokenized_corpus = word_tokenize(text)
    print('end------tokenized_corpus')

    # 现在，tokenized_corpus 是一个包含分词后单词的列表
    # print(tokenized_corpus)
    return tokenized_corpus

@memory.cache
def extract_shared(datas):
    tokenized_corpus = tokenize(datas)
    # 2. Extract trivially shared n-grams
    k = 500
    # <tokenized_corpus> is a list of strings
    # Extract all n-grams of length 1-4
    all_ngrams = []
    print('start------trivially_shared_ngrams')
    for n in range(1, 5):
        all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
    # Calculate frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = set(frequencies.most_common(k))
    print('end------trivially_shared_ngrams')
    return trivially_shared_ngrams

def get_ref_can(file_path_can, file_path_ref):
    cans = []
    refs = []
    with jsonlines.open(file_path_can,'r') as f:
        for dat in f:
            cans.append(dat)
    with jsonlines.open(file_path_ref,'r') as ff:
        for datt in ff:
            refs.append(datt)
    return cans, refs

def calculate_crystal_bleu(references, candidates, trivially_shared_ngrams):
    # 这里使用你提供的corpus_bleu函数计算CrystalBLEU得分
    # crystalBLEU_score = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
    crystalBLEU_score = corpus_bleu(
        references,
        candidates,
        weights=(0.5, 0.5), # (0.25, 0.25, 0.25, 0.25)
        ignoring=trivially_shared_ngrams
    )
    # crystalBLEU_score = sentence_bleu(references, candidates, ignoring=trivially_shared_ngrams)
    return crystalBLEU_score

@cached(cache=LRUCache(maxsize=128))
def clean_and_tokenize(s: str):
    s = clean_str(s)
    s_tokens = s.split(" ")
    return s_tokens

def calculate_bleu(refs, cans, trivially_shared_ngrams, i: int, j: int):
    can = cans[i]
    code_can = clean_and_tokenize(can['code'])

    ref = refs[j]
    code_ref = clean_and_tokenize(ref['code'])

    crystalBLEU_score = calculate_crystal_bleu([[code_ref]], [code_can], trivially_shared_ngrams)
    return crystalBLEU_score

def bleu_parallel(cans, refs, trivially_shared_ngrams, i):
    best_score = 0.0
    best_ref = -1

    ret = []
    for j in range(len(refs)):
        # 计算CrystalBLEU得分
        crystalBLEU_score = calculate_bleu(refs, cans, trivially_shared_ngrams, i, j)
        ret.append(crystalBLEU_score)

    for j, crystalBLEU_score in enumerate(ret):
        # 更新最高得分和对应的参考代码
        if crystalBLEU_score > best_score:
            best_score = crystalBLEU_score
            best_ref = j
    
    return best_score, best_ref


def calculate(trivially_shared_ngrams, cans, refs, name, datas, num):
    print("Start bleu")
    
    ret = Parallel(n_jobs=8, backend="loky", pre_dispatch = '2 * n_jobs')(
        delayed(bleu_parallel)(
            [None] * i + [cans[i]] + [None] * (len(cans) - i - 1),
            refs,
            trivially_shared_ngrams,
            i
        )
        for i in tqdm(range(len(cans)))
    )
    """
    ret = []
    for i in tqdm(range(len(cans))):
        ret.append(bleu_parallel(cans, refs, trivially_shared_ngrams, i))
    """

    with jsonlines.open(f'similar_result/{name}_output_{num}.jsonl', 'w') as fp:
        for best_score, best_ref in tqdm(ret):
            try:
                # 将最高得分和对应的参考代码写入文件
                fp.write({
                    "best_score": best_score,
                    "best_ref": best_ref,
                    "best_data": datas[best_ref],
                })
            except:
                print(best_score,best_ref)
                print(len(datas))



def main():
    name = 'what'
    num2 = 20000
    num1 = 0
    file_path_what = '/home/zxw/llm/similar_shot/sample_train/what_train_sample.jsonl'
    database_what = get_database(file_path_what)
    print(len(database_what))
    # print(database_what[0:1])
    # database_what = sample(database_what_all)
    trivially_shared_ngrams = extract_shared(database_what)

    file_path_can = '/home/zxw/llm/generate_comment/what_sample.jsonl'
    file_path_ref = '/home/zxw/llm/similar_shot/sample_train/what_train_sample.jsonl'
    cans, refs = get_ref_can(file_path_can, file_path_ref)
    # calculate(trivially_shared_ngrams, cans, refs, name, database_how)
    # print(cans[num1:num2])
    calculate(trivially_shared_ngrams, cans[num1:num2], refs, name, database_what, num2)

if __name__ =='__main__':
    main()


# import jsonlines
# import numpy as np
# import random

# from nltk.util import ngrams
# # 1. Import CrystalBLEU
# from crystalbleu import corpus_bleu, sentence_bleu

# import re
# from nltk.tokenize import word_tokenize
# from tqdm import tqdm

# from joblib import Parallel, delayed
# from functools import lru_cache
# from cachetools import cached, LRUCache, TTLCache
# from cachetools.keys import hashkey
# import multiprocessing

# from joblib import Memory
# location = '~/.cache/crystal_bleu'
# memory = Memory(location, verbose=0, compress=True)

# """
# cython  test_crystalBLEU.py
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.5 -o test_crystalBLEU.so test_crystalBLEU.c
# """


# """
# Faster BLEU
# """

# import math
# import sys
# import warnings
# from collections import Counter, deque
# from fractions import Fraction


# class SmoothingFunction:
#     """
#     This is an implementation of the smoothing techniques
#     for segment-level BLEU scores that was presented in
#     Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
#     Smoothing Techniques for Sentence-Level BLEU. In WMT14.
#     http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
#     """

#     def __init__(self, epsilon=0.1, alpha=5, k=5):
#         """
#         This will initialize the parameters required for the various smoothing
#         techniques, the default values are set to the numbers used in the
#         experiments from Chen and Cherry (2014).

#         >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
#         ...                 'that', 'the', 'military', 'always', 'obeys', 'the',
#         ...                 'commands', 'of', 'the', 'party']
#         >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
#         ...               'that', 'the', 'military', 'will', 'forever', 'heed',
#         ...               'Party', 'commands']

#         >>> chencherry = SmoothingFunction()
#         >>> print(sentence_bleu([reference1], hypothesis1)) # doctest: +ELLIPSIS
#         0.4118...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method0)) # doctest: +ELLIPSIS
#         0.4118...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method1)) # doctest: +ELLIPSIS
#         0.4118...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method2)) # doctest: +ELLIPSIS
#         0.4489...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method3)) # doctest: +ELLIPSIS
#         0.4118...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method4)) # doctest: +ELLIPSIS
#         0.4118...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method5)) # doctest: +ELLIPSIS
#         0.4905...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method6)) # doctest: +ELLIPSIS
#         0.4135...
#         >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method7)) # doctest: +ELLIPSIS
#         0.4905...

#         :param epsilon: the epsilon value use in method 1
#         :type epsilon: float
#         :param alpha: the alpha value use in method 6
#         :type alpha: int
#         :param k: the k value use in method 4
#         :type k: int
#         """
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.k = k

#     def method0(self, p_n, *args, **kwargs):
#         """
#         No smoothing.
#         """
#         p_n_new = []
#         for i, p_i in enumerate(p_n):
#             if p_i.numerator != 0:
#                 p_n_new.append(p_i)
#             else:
#                 # _msg = str(
#                 #     "\nThe hypothesis contains 0 counts of {}-gram overlaps.\n"
#                 #     "Therefore the BLEU score evaluates to 0, independently of\n"
#                 #     "how many N-gram overlaps of lower order it contains.\n"
#                 #     "Consider using lower n-gram order or use "
#                 #     "SmoothingFunction()"
#                 # ).format(i + 1)
#                 # warnings.warn(_msg)

#                 # When numerator==0 where denonminator==0 or !=0, the result
#                 # for the precision score should be equal to 0 or undefined.
#                 # Due to BLEU geometric mean computation in logarithm space,
#                 # we we need to take the return sys.float_info.min such that
#                 # math.log(sys.float_info.min) returns a 0 precision score.
#                 p_n_new.append(sys.float_info.min)
#         return p_n_new

#     def method1(self, p_n, *args, **kwargs):
#         """
#         Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
#         """
#         return [
#             (p_i.numerator + self.epsilon) / p_i.denominator
#             if p_i.numerator == 0
#             else p_i
#             for p_i in p_n
#         ]

#     def method2(self, p_n, *args, **kwargs):
#         """
#         Smoothing method 2: Add 1 to both numerator and denominator from
#         Chin-Yew Lin and Franz Josef Och (2004) ORANGE: a Method for
#         Evaluating Automatic Evaluation Metrics for Machine Translation.
#         In COLING 2004.
#         """
#         return [
#             Fraction(p_n[i].numerator + 1,
#                      p_n[i].denominator + 1, _normalize=False)
#             if i != 0 else p_n[0]
#             for i in range(len(p_n))
#         ]

#     def method3(self, p_n, *args, **kwargs):
#         """
#         Smoothing method 3: NIST geometric sequence smoothing
#         The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
#         precision score whose matching n-gram count is null.
#         k is 1 for the first 'n' value for which the n-gram match count is null/
#         For example, if the text contains:
#          - one 2-gram match
#          - and (consequently) two 1-gram matches
#         the n-gram count for each individual precision score would be:
#          - n=1  =>  prec_count = 2     (two unigrams)
#          - n=2  =>  prec_count = 1     (one bigram)
#          - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
#          - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
#         """
#         incvnt = 1  # From the mteval-v13a.pl, it's referred to as k.
#         for i, p_i in enumerate(p_n):
#             if p_i.numerator == 0:
#                 p_n[i] = 1 / (2 ** incvnt * p_i.denominator)
#                 incvnt += 1
#         return p_n

#     def method4(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
#         """
#         Smoothing method 4:
#         Shorter translations may have inflated precision values due to having
#         smaller denominators; therefore, we give them proportionally
#         smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
#         suggests dividing by 1/ln(len(T)), where T is the length of the translation.
#         """
#         incvnt = 1
#         hyp_len = hyp_len if hyp_len else len(hypothesis)
#         for i, p_i in enumerate(p_n):
#             if p_i.numerator == 0 and hyp_len > 1:
#                 #                 incvnt = i + 1 * self.k / math.log(
#                 #                     hyp_len
#                 #                 )  # Note that this K is different from the K from NIST.
#                 #                 p_n[i] = incvnt / p_i.denominator\
#                 numerator = 1 / (2 ** incvnt * self.k / math.log(hyp_len))
#                 p_n[i] = numerator / p_i.denominator
#                 incvnt += 1
#         return p_n

#     def method5(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
#         """
#         Smoothing method 5:
#         The matched counts for similar values of n should be similar. To a
#         calculate the n-gram matched count, it averages the n−1, n and n+1 gram
#         matched counts.
#         """
#         hyp_len = hyp_len if hyp_len else len(hypothesis)
#         m = {}
#         # Requires an precision value for an addition ngram order.
#         p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
#         m[-1] = p_n[0] + 1
#         for i, p_i in enumerate(p_n):
#             p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
#             m[i] = p_n[i]
#         return p_n

#     def method6(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
#         """
#         Smoothing method 6:
#         Interpolates the maximum likelihood estimate of the precision *p_n* with
#         a prior estimate *pi0*. The prior is estimated by assuming that the ratio
#         between pn and pn−1 will be the same as that between pn−1 and pn−2; from
#         Gao and He (2013) Training MRF-Based Phrase Translation Models using
#         Gradient Ascent. In NAACL.
#         """
#         hyp_len = hyp_len if hyp_len else len(hypothesis)
#         # This smoothing only works when p_1 and p_2 is non-zero.
#         # Raise an error with an appropriate message when the input is too short
#         # to use this smoothing technique.
#         assert p_n[2], "This smoothing method requires non-zero precision for bigrams."
#         for i, p_i in enumerate(p_n):
#             if i in [0, 1]:  # Skips the first 2 orders of ngrams.
#                 continue
#             else:
#                 pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
#                 # No. of ngrams in translation that matches the reference.
#                 m = p_i.numerator
#                 # No. of ngrams in translation.
#                 l = sum(1 for _ in ngrams(hypothesis, i + 1))
#                 # Calculates the interpolated precision.
#                 p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
#         return p_n

#     def method7(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
#         """
#         Smoothing method 7:
#         Interpolates methods 4 and 5.
#         """
#         hyp_len = hyp_len if hyp_len else len(hypothesis)
#         p_n = self.method4(p_n, references, hypothesis, hyp_len)
#         p_n = self.method5(p_n, references, hypothesis, hyp_len)
#         return p_n

# def closest_ref_length(references, hyp_len):
#     ref_lens = (len(reference) for reference in references)
#     closest_ref_len = min(
#         ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
#     )
#     return closest_ref_len

# @cached(cache=LRUCache(maxsize=128))
# def get_ngrams(sequence, n):
#     """
#     result = []
#     for i in range(len(sequence) - n + 1):
#         ngram = sequence[i:i+n]
#         result.append(tuple(ngram))
#     return result
#     """
#     all_ngrams = ngrams(sequence, n)
#     return all_ngrams

# def ngrams_ignoring(sequence, n, ignoring=None):
#     all_ngrams = get_ngrams(sequence, n)
#     if ignoring == None: # Change to "if True:" if you want the weighted approach 
#         return all_ngrams
#     ret = []
#     for item in all_ngrams:
#         item = tuple(item)
#         if item not in ignoring:
#             ret.append(item)
#     return ret

# def brevity_penalty(closest_ref_len, hyp_len):
#     if hyp_len > closest_ref_len:
#         return 1
#     # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
#     elif hyp_len == 0:
#         return 0
#     else:
#         return math.exp(1 - closest_ref_len / hyp_len)

# def modified_precision(references, hypothesis, n, ignoring=None, list_of_references_ngrams=None, hypotheses_ngrams=None):
#     # Extracts all ngrams in hypothesis
#     # Set an empty Counter if hypothesis is empty.
#     if len(hypothesis) >= n:
#         if hypotheses_ngrams is not None:
#             counts = hypotheses_ngrams[n-1]
#         else:
#             counts = Counter(ngrams_ignoring(tuple(hypothesis), n, ignoring=ignoring))
#     else:
#         counts = Counter()
#     # Extract a union of references' counts.
#     # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
#     max_counts = {}
#     for ref_i, reference in enumerate(references):
#         if len(reference) >= n:
#             if list_of_references_ngrams is not None:
#                 reference_counts = list_of_references_ngrams[ref_i][n-1]
#             else:
#                 reference_counts = Counter(ngrams_ignoring(tuple(reference), n, ignoring=ignoring))
#         else:
#             reference_counts = Counter()
#         for ngram in counts:
#             max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
    
#     # Uncomment if you want to use the weighted approach
#     # if ignoring:
#     #     for k, v in counts.items():
#     #         if (k in ignoring) and (ignoring[k] > 1):
#     #             counts[k] /= math.log(ignoring[k])
#     #             max_counts[k] /= math.log(ignoring[k])

#     # Assigns the intersection between hypothesis and references' counts.
#     clipped_counts = {
#         ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
#     }

#     numerator = int(sum(clipped_counts.values()))
#     # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
#     # Usually this happens when the ngram order is > len(reference).
#     denominator = int(max(1, sum(counts.values())))

#     return Fraction(numerator, denominator, _normalize=False)

# def corpus_bleu(
#     list_of_references,
#     hypotheses,
#     list_of_references_ngrams=None,
#     hypotheses_ngrams=None,
#     weights=(0.25, 0.25, 0.25, 0.25),
#     smoothing_function=None,
#     auto_reweigh=False,
#     ignoring=None,
# ):
#     # Before proceeding to compute BLEU, perform sanity checks.

#     # Key = ngram order, and value = no. of ngram matches.
#     p_numerators = Counter()
#     # Key = ngram order, and value = no. of ngram in ref.
#     p_denominators = Counter()
#     hyp_lengths, ref_lengths = 0, 0

#     assert len(list_of_references) == len(hypotheses), (
#         "The number of hypotheses and their reference(s) should be the " "same "
#     )

#     # Iterate through each hypothesis and their corresponding references.
#     for references, hypothesis in zip(list_of_references, hypotheses):
#         # For each order of ngram, calculate the numerator and
#         # denominator for the corpus-level modified precision.
#         for i, _ in enumerate(weights, start=1):
#             p_i = modified_precision(references, hypothesis, i, ignoring=ignoring, list_of_references_ngrams=list_of_references_ngrams, hypotheses_ngrams=hypotheses_ngrams)
#             p_numerators[i] += p_i.numerator
#             p_denominators[i] += p_i.denominator

#         # Calculate the hypothesis length and the closest reference length.
#         # Adds them to the corpus-level hypothesis and reference counts.
#         hyp_len = len(hypothesis)
#         hyp_lengths += hyp_len
#         ref_lengths += closest_ref_length(references, hyp_len)

#     # Calculate corpus-level brevity penalty.
#     bp = brevity_penalty(ref_lengths, hyp_lengths)

#     # Uniformly re-weighting based on maximum hypothesis lengths if largest
#     # order of n-grams < 4 and weights is set at default.
#     if auto_reweigh:
#         if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
#             weights = (1 / hyp_lengths,) * hyp_lengths

#     # Collects the various precision values for the different ngram orders.
#     p_n = [
#         Fraction(p_numerators[i], p_denominators[i], _normalize=False)
#         for i, _ in enumerate(weights, start=1)
#     ]

#     # Returns 0 if there's no matching n-grams
#     # We only need to check for p_numerators[1] == 0, since if there's
#     # no unigrams, there won't be any higher order ngrams.
#     if p_numerators[1] == 0:
#         return 0

#     # If there's no smoothing, set use method0 from SmoothinFunction class.
#     if not smoothing_function:
#         smoothing_function = SmoothingFunction().method0
#     # Smoothen the modified precision.
#     # Note: smoothing_function() may convert values into floats;
#     #       it tries to retain the Fraction object as much as the
#     #       smoothing method allows.
#     p_n = smoothing_function(
#         p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
#     )
#     s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
#     s = bp * math.exp(math.fsum(s))
#     return s

# """
# End of fast BLEU
# """

# @memory.cache
# def get_database(file_path_how):
#     datas = []
#     with jsonlines.open(file_path_how, 'r')as f:
#         for dat in f:
#             datas.append(dat)
    
#     return datas

# # def read_data(file_address):
# #     # ids = []
# #     codes = []
# #     comments = []
# #     # labels = []
# #     with jsonlines.open(file_address, "r") as file:
# #         lines=[]
# #         for line in file:
# #             lines.append(line)
# #     for line in lines:
# #         # sample = json.loads(line)
# #         # ids.append(sample.get("id"))
# #         codes.append(line["code"])
# #         comments.append(line["comment"])
# #         # labels.append(sample.get("label"))
# #     # print('Number of samples is ' + str(len(ids)))
# #     print('Number of codes is ' + str(len(codes)))
# #     print('Number of comments is ' + str(len(comments)))
# #     # print('Number of labels is ' + str(len(labels)))
# #     return codes, comments
# # def get_data():
# #     training_codes, training_comments, test_codes, test_comments = [], [], [], []

# #     train_file_address = '/home/zxw/llm/dataset_base/how_train.jsonl'
# #     # train_file_address = 'tlcodesum.train'
# #     cur_codes, cur_comments = read_data(train_file_address)
# #     training_codes += cur_codes
# #     training_comments += cur_comments
# #     # training_labels += cur_labels
    
# #     test_file_address = '/home/zxw/llm/generate_comment/how_sample.jsonl'
# #     # test_file_address = 'tlcodesum.test'
# #     cur_codes, cur_comments = read_data(test_file_address)
# #     test_codes += cur_codes
# #     test_comments += cur_comments
# #     # test_labels += cur_labels
# #     return training_codes, training_comments, test_codes, test_comments, 


# # def tokenize(code_str):

# #     code_str = str(code_str)
# #     code_str = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_str)
# #     code_str = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_str)
# #     code_str = re.sub(r'\s+', ' ', code_str)
# #     tokens = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+|[^\w\s]+', code_str)
# #     for i in range(len(tokens)):
# #         if i > 0 and tokens[i-1].islower() and tokens[i].isupper():
# #             tokens[i] = tokens[i].lower()
# #     return tokens

# def clean_str(s: str) -> str:
#     s = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', s)
#     s = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', s)
#     s = re.sub(r'\s+', ' ', s)
#     return s

# def tokenize(datas):
#     codes = []
#     for dat in datas:
#         code_str = dat['code']
#         code_str = clean_str(code_str)
#         codes.append(code_str)
#     text = " ".join(codes)
#     # text = " ".join(datas)
#     # 假设你有一段文本
#     # text = "This is a sample sentence."
#     print('start------tokenized_corpus')
#     # 对文本进行分词
#     tokenized_corpus = word_tokenize(text)
#     print('end------tokenized_corpus')

#     # 现在，tokenized_corpus 是一个包含分词后单词的列表
#     # print(tokenized_corpus)
#     return tokenized_corpus

# @memory.cache
# def extract_shared(datas):
#     tokenized_corpus = tokenize(datas)
#     # 2. Extract trivially shared n-grams
#     k = 500
#     # <tokenized_corpus> is a list of strings
#     # Extract all n-grams of length 1-4
#     all_ngrams = []
#     print('start------trivially_shared_ngrams')
#     for n in range(1, 5):
#         all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
#     # Calculate frequencies of all n-grams
#     frequencies = Counter(all_ngrams)
#     trivially_shared_ngrams = set(frequencies.most_common(k))
#     print('end------trivially_shared_ngrams')
#     return trivially_shared_ngrams

# def get_ref_can(file_path_can, file_path_ref):
#     cans = []
#     refs = []
#     with jsonlines.open(file_path_can,'r') as f:
#         for dat in f:
#             cans.append(dat)
#     with jsonlines.open(file_path_ref,'r') as ff:
#         for datt in ff:
#             refs.append(datt)
#     return cans, refs
# # def calculate(trivially_shared_ngrams,cans,refs):
# #     codes_can = []
# #     codes_ref = []
# #     for dat in cans:
# #         code = dat['code']
# #         codes_can.append(code)
# #     for dat in refs:
# #         code = dat['code']
# #         codes_ref.append(code)


# def calculate_crystal_bleu(references, candidates, trivially_shared_ngrams, list_of_references_ngrams=None, hypotheses_ngrams=None):
#     # 这里使用你提供的corpus_bleu函数计算CrystalBLEU得分
#     # crystalBLEU_score = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
#     crystalBLEU_score = corpus_bleu(
#         references,
#         candidates,
#         weights=(0.25, 0.25, 0.25, 0.25),
#         ignoring=trivially_shared_ngrams,
#         list_of_references_ngrams=list_of_references_ngrams,
#         hypotheses_ngrams=hypotheses_ngrams
#     )
#     # crystalBLEU_score = sentence_bleu(references, candidates, ignoring=trivially_shared_ngrams)
#     return crystalBLEU_score

# @cached(cache=LRUCache(maxsize=128))
# def clean_and_tokenize(s: str):
#     s = clean_str(s)
#     s_tokens = s.split(" ")
#     return s_tokens

# def calculate_bleu(refs, cans, trivially_shared_ngrams, i: int, j: int, refs_ngrams=None, cans_ngrams=None):
#     can = cans[i]
#     code_can = clean_and_tokenize(can['code'])

#     ref = refs[j]
#     code_ref = clean_and_tokenize(ref['code'])

#     if refs_ngrams is None:
#         refs_ngrams_param = None
#     else:
#         # print('yes--------refs_ngrams')
#         refs_ngrams_param = [refs_ngrams]
    
#     if cans_ngrams is None:
#         cans_ngrams_param = None
#     else:
#         # print('yes--------cans_ngrams')
#         cans_ngrams_param = cans_ngrams
#     crystalBLEU_score = calculate_crystal_bleu([[code_ref]], [code_can], trivially_shared_ngrams, refs_ngrams_param, cans_ngrams_param)
#     return crystalBLEU_score

# def bleu_parallel(cans, refs, trivially_shared_ngrams, i, cans_ngrams, refs_ngrams):
#     best_score = 0.0
#     best_ref = -1

#     ret = []
#     for j in range(len(refs)):
#         # 计算CrystalBLEU得分
#         crystalBLEU_score = calculate_bleu(refs, cans, trivially_shared_ngrams, i, j,  refs_ngrams[j], cans_ngrams[i])
#         ret.append(crystalBLEU_score)
#     print('start----------count--------best')
#     for j, crystalBLEU_score in enumerate(ret):
#         # 更新最高得分和对应的参考代码
#         if crystalBLEU_score > best_score:
#             best_score = crystalBLEU_score
#             best_ref = j
#     print('end------------print----------best')
#     return best_score, best_ref


# def calculate(trivially_shared_ngrams, cans, refs, name, datas ,num):
#     print("Start bleu")
#     # ret = Parallel(n_jobs=16, backend="loky")(delayed(bleu_parallel)([cans[i]], refs, trivially_shared_ngrams, 0, [cans_ngrams[0]], refs_ngrams) for i in tqdm(range(len(cans))))
#     cans_ngrams = []
#     for can in tqdm(cans):
#         code_can = clean_and_tokenize(can['code'])
#         item = []
#         for n in range(1, 5):
#             item.append(Counter(ngrams_ignoring(tuple(code_can), n, ignoring=trivially_shared_ngrams)))
#         cans_ngrams.append(item)

#     refs_ngrams = []
#     for ref in tqdm(refs):
#         code_ref = clean_and_tokenize(ref['code'])
#         item = []
#         for n in range(1, 5):
#             item.append(Counter(ngrams_ignoring(tuple(code_ref), n, ignoring=trivially_shared_ngrams)))
#         refs_ngrams.append(item)
#     ret = Parallel(n_jobs=16, backend="loky", pre_dispatch = '2 * n_jobs')(delayed(bleu_parallel)([cans[i]], refs, trivially_shared_ngrams, 0, [cans_ngrams[0]], refs_ngrams) for i in tqdm(range(len(cans))))

#     # ret = []
#     # for i in tqdm(range(len(cans))):
#     #     ret.append(bleu_parallel(cans, refs, trivially_shared_ngrams, i, cans_ngrams, refs_ngrams))

#     with jsonlines.open(f'similar_result/{name}_output_{num}.jsonl', 'w') as fp:
#         for best_score, best_ref in ret:
#             print(best_score, best_ref)
#             # 将最高得分和对应的参考代码写入文件
#             fp.write({
#                 "best_score": best_score,
#                 "best_ref": best_ref,
#                 "best_data": datas[best_ref],
#             })


# def sample(data):
#     print('start------------code--------75')
#     codes = []
#     # 从 data 中提取 'code' 字段并存储为新的列表 codes
#     codes = [item['code'] for item in data]
    
#     # 使用 NLTK 分词，计算 codes 列表中元素的 token 数目
#     token_counts = [len(word_tokenize(code)) for code in codes]
    
#     # 使用 NumPy 计算四分之三位值
#     percentile_75 = np.percentile(token_counts, 75)
    
#     # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
#     new_data_list = [data[i] for i, token_count in enumerate(token_counts) if token_count < percentile_75]

#     print('start------------comment--------75')

#     comments = []
#     # 从 new_data_list 中提取 'comment' 字段并存储为新的列表 comments
#     comments = [item['comment'] for item in new_data_list]
    
#     # 使用 NLTK 分词，计算 comments 列表中元素的 token 数目
#     token_counts1 = [len(word_tokenize(comment)) for comment in comments]
    
#     # 使用 NumPy 计算四分之三位值
#     percentile_75_1 = np.percentile(token_counts1, 75)
    
#     # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
#     new_data_list1 = [new_data_list[i] for i, token_count in enumerate(token_counts1) if token_count < percentile_75_1]
#     print(len(new_data_list1))
#     # final_list = random.sample(new_data_list1,20000)
#     return new_data_list1
# def main():
#     name = 'what'
#     num2 = 20
#     num1 = 0
#     file_path_what = '/home/zxw/llm/dataset_base/what_train.jsonl'
#     database_what_all = get_database(file_path_what)
#     database_what = sample(database_what_all)
#     trivially_shared_ngrams = extract_shared(database_what)

#     file_path_can = '/home/zxw/llm/generate_comment/what_sample.jsonl'
#     file_path_ref = '/home/zxw/llm/dataset_base/what_train.jsonl'
#     cans, refs = get_ref_can(file_path_can, file_path_ref)
#     # calculate(trivially_shared_ngrams, cans, refs, name, database_how)
#     # print(cans[num1:num2])
#     calculate(trivially_shared_ngrams, cans[num1:num2], refs, name, database_what, num2)

# if __name__ =='__main__':
#     main()





# def main1():
#     name = 'how'
#     num2 = [5000,10000,15000,2000]
#     num1 = [0,5000,10000,15000]
#     # file_path_how = './dataset_base/how_train.jsonl'
#     file_path_how = '/home/zxw/llm/dataset_base/how_train.jsonl'

#     database_how = get_database(file_path_how)
#     trivially_shared_ngrams = extract_shared(database_how)

#     # file_path_can = './generate_comment/how_sample.jsonl'
#     # file_path_ref = './dataset_base/how_train.jsonl'
#     file_path_can = '/home/zxw/llm/generate_comment/how_sample.jsonl'
#     file_path_ref = '/home/zxw/llm/dataset_base/how_train.jsonl'
#     cans, refs = get_ref_can(file_path_can, file_path_ref)
#     for i in range(len(num1)):
#         calculate(trivially_shared_ngrams, cans[num1[i]:num2[i]], refs, name, database_how, num2[i])

        

# def main2():
#     name = 'what'
#     # file_path_how = './dataset_base/how_train.jsonl'
#     file_path_what = '/home/zxw/llm/dataset_base/what_train.jsonl'

#     database_what = get_database(file_path_what)
#     trivially_shared_ngrams = extract_shared(database_what)

#     # file_path_can = './generate_comment/how_sample.jsonl'
#     # file_path_ref = './dataset_base/how_train.jsonl'
#     file_path_can = '/home/zxw/llm/generate_comment/what_sample.jsonl'
#     file_path_ref = '/home/zxw/llm/dataset_base/how_train.jsonl'
#     cans, refs = get_ref_can(file_path_can, file_path_ref)
#     calculate(trivially_shared_ngrams, cans, refs, name, database_what)

# def main3():
#     name = 'why'
#     # file_path_how = './dataset_base/how_train.jsonl'
#     file_path_why = '/home/zxw/llm/dataset_base/why_train.jsonl'

#     database_why = get_database(file_path_why)
#     trivially_shared_ngrams = extract_shared(database_why)

#     # file_path_can = './generate_comment/how_sample.jsonl'
#     # file_path_ref = './dataset_base/how_train.jsonl'
#     file_path_can = '/home/zxw/llm/generate_comment/why_sample.jsonl'
#     file_path_ref = '/home/zxw/llm/dataset_base/why_train.jsonl'
#     cans, refs = get_ref_can(file_path_can, file_path_ref)
#     calculate(trivially_shared_ngrams, cans, refs, name, database_why)





# if __name__ =='__main__':
#     main1()

    # main2()
    
    # main3()

# from nltk.translate.bleu_score import sentence_bleu

# reference = ["Build the polynomials by iterating on the top diagonal of the divided differences array"]
# candidate = " Update polynomials using the top diagonal elements and coefficients"

# # Convert the reference to a list of lists (each word is a list)
# reference = [ref.split() for ref in reference]
# # Convert the candidate to a list of words
# candidate = candidate.split()

# # Calculate BLEU score
# bleu_score = sentence_bleu(reference, candidate)
# print("BLEU Score:", bleu_score)


















# from collections import Counter
# from nltk.util import ngrams
# # 1. Import CrystalBLEU
# from crystalbleu import corpus_bleu
# import jsonlines

# import re
# from nltk.tokenize import word_tokenize
# from tqdm import tqdm
# def get_database(file_path_how):
#     datas = []
#     with jsonlines.open(file_path_how,'r')as f:
#         for dat in f:
#             datas.append(dat)
    
#     return datas

# def tokenize(datas):
#     codes = []
#     for dat in datas:
#         code_str = dat['code']
#         code_str = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_str)
#         code_str = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_str)
#         code_str = re.sub(r'\s+', ' ', code_str)
#         codes.append(code_str)
#     text = " ".join(codes)    

#     print('start------tokenized_corpus')
#     # 对文本进行分词
#     tokenized_corpus = word_tokenize(text)
#     print('end------tokenized_corpus')

#     # tokenized_corpus 是一个包含分词后单词的列表

#     return tokenized_corpus

# def extract_shared(datas):
#     tokenized_corpus = tokenize(datas)
#     # 2. Extract trivially shared n-grams
#     k = 500
#     # <tokenized_corpus> is a list of strings
#     # Extract all n-grams of length 1-4
#     all_ngrams = []
#     print('start------trivially_shared_ngrams')
#     for n in range(1, 5):
#         all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
#     # Calculate frequencies of all n-grams
#     frequencies = Counter(all_ngrams)
#     trivially_shared_ngrams = dict(frequencies.most_common(k))
#     print('end------trivially_shared_ngrams')
#     return trivially_shared_ngrams
# def get_ref_can(file_path_can,file_path_ref):
#     cans = []
#     refs = []
#     with jsonlines.open(file_path_can,'r') as f:
#         for dat in f:
#             cans.append(dat)
#     with jsonlines.open(file_path_ref,'r') as ff:
#         for datt in ff:
#             refs.append(datt)
#     return cans, refs

        

# def calculate(trivially_shared_ngrams, cans, refs, name, datas):
#     with jsonlines.open(f'/home/zxw/llm/similar_shot/similar_result/{name}_output.jsonl', 'w') as fp:
#         for can in tqdm(cans):
#             code_can = can['code']
#             code_can = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_can)
#             code_can = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_can)
#             code_can = re.sub(r'\s+', ' ', code_can)
#             best_score = 0.0
#             best_ref = ""

#             for i in range(len(refs)):
#                 ref = refs[i]
#                 code_ref = ref['code']
#                 code_ref = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_ref)
#                 code_ref = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_ref)
#                 code_ref = re.sub(r'\s+', ' ', code_ref)
#                 # 计算CrystalBLEU得分
#                 crystalBLEU_score = calculate_crystal_bleu([[code_ref]], [code_can], trivially_shared_ngrams)

#                 # 更新最高得分和对应的参考代码
#                 if crystalBLEU_score > best_score:
#                     best_score = crystalBLEU_score
#                     best_ref = datas[i]
#                 # if best_score > 0.5:
#                 #     break
#                 print(best_score)
#             # 将最高得分和对应的数据写入文件
#             fp.write(best_ref)

# def calculate_crystal_bleu(references, candidates, trivially_shared_ngrams):
#     # 使用corpus_bleu函数计算CrystalBLEU得分
#     crystalBLEU_score = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
#     return crystalBLEU_score



# if __name__ =='__main__':
    
    
#     name = 'how'
#     file_path_how = '/home/zxw/llm/dataset_base/how_train.jsonl'
#     database_how = get_database(file_path_how)
#     trivially_shared_ngrams =  extract_shared(database_how)


#     file_path_can = '/home/zxw/llm/generate_comment/how_sample.jsonl'
#     file_path_ref = '/home/zxw/llm/dataset_base/how_train.jsonl'
#     cans, refs = get_ref_can(file_path_can, file_path_ref)
#     calculate(trivially_shared_ngrams, cans, refs,name, database_how)


