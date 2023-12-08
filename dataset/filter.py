import nltk
import jsonlines
import numpy as np
import random
# 导入nltk的分词工具
from nltk.tokenize import word_tokenize

def read_file():
    how_data = []
    why_data = []
    what_data = []
    how_file_path = 'D:\\Research\\Inline_generation\\inlinebart\\dataset\\how_test.jsonl'
    why_file_path = 'D:\\Research\\Inline_generation\\inlinebart\\dataset\\why_test.jsonl'
    what_file_path = 'D:\\Research\\Inline_generation\\inlinebart\\dataset\\what_test.jsonl'
    print('start----how-------')
    with jsonlines.open(how_file_path,'r')as f:
        for dat in f:
            how_data.append(dat)
    print('start----why-------')
    with jsonlines.open(why_file_path,'r')as ff:
        for dat in ff:
            why_data.append(dat)
    print('start----what-------')
    with jsonlines.open(what_file_path,'r')as fff:
        for dat in fff:
            what_data.append(dat)
    return how_data, why_data, what_data            

def sample(data):
    codes = []
    # 从 data 中提取 'code' 字段并存储为新的列表 codes
    codes = [item['code'] for item in data]
    
    # 使用 NLTK 分词，计算 codes 列表中元素的 token 数目
    token_counts = [len(word_tokenize(code)) for code in codes]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75 = np.percentile(token_counts, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list = [data[i] for i, token_count in enumerate(token_counts) if token_count < percentile_75]


    comments = []
    # 从 new_data_list 中提取 'comment' 字段并存储为新的列表 comments
    comments = [item['comment'] for item in new_data_list]
    
    # 使用 NLTK 分词，计算 comments 列表中元素的 token 数目
    token_counts1 = [len(word_tokenize(comment)) for comment in comments]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75_1 = np.percentile(token_counts1, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list1 = [new_data_list[i] for i, token_count in enumerate(token_counts1) if token_count < percentile_75_1]

    final_list = random.sample(new_data_list1,20000)
    return final_list

if __name__ == '__main__':
    how_data, why_data, what_data = read_file()
    how_sample = sample(how_data)
    why_sample = sample(why_data)
    what_sample = sample(what_data)
    with jsonlines.open ('how_sample.jsonl','w')as writer1:
        for dat in how_sample:
            writer1.write(dat)
    
    with jsonlines.open ('why_sample.jsonl','w')as writer2:
        for dat in why_sample:
            writer2.write(dat)
        
    with jsonlines.open ('what_sample.jsonl','w')as writer3:
        for dat in what_sample:
            writer3.write(dat)
        
    
    
    