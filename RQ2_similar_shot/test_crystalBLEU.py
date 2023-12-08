
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
    name = 'how'
    num2 = 20000
    num1 = 0
    file_path_how = '/home/zxw/llm/similar_shot/sample_train/how_train_sample.jsonl'
    database_how = get_database(file_path_how)
    # database_how = sample(database_how_all)
    trivially_shared_ngrams = extract_shared(database_how)

    file_path_can = '/home/zxw/llm/generate_comment/how_sample.jsonl'
    file_path_ref = '/home/zxw/llm/similar_shot/sample_train/how_train_sample.jsonl'
    cans, refs = get_ref_can(file_path_can, file_path_ref)
    # calculate(trivially_shared_ngrams, cans, refs, name, database_how)
    # print(cans[num1:num2])
    calculate(trivially_shared_ngrams, cans[num1:num2], refs, name, database_how, num2)

if __name__ =='__main__':
    main()





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


