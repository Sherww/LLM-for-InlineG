

import numpy as np

import jsonlines
from nltk.tokenize import word_tokenize

def get_database(file_path_how):
    datas = []
    with jsonlines.open(file_path_how, 'r')as f:
        for dat in f:
            datas.append(dat)
    
    return datas



def sample(data):
    print('start------------code--------75')
    codes = []
    # 从 data 中提取 'code' 字段并存储为新的列表 codes
    codes = [item['code'] for item in data]
    
    # 使用 NLTK 分词，计算 codes 列表中元素的 token 数目
    token_counts = [len(word_tokenize(code)) for code in codes]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75 = np.percentile(token_counts, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list = [data[i] for i, token_count in enumerate(token_counts) if token_count < percentile_75]

    print('start------------comment--------75')

    comments = []
    # 从 new_data_list 中提取 'comment' 字段并存储为新的列表 comments
    comments = [item['comment'] for item in new_data_list]
    
    # 使用 NLTK 分词，计算 comments 列表中元素的 token 数目
    token_counts1 = [len(word_tokenize(comment)) for comment in comments]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75_1 = np.percentile(token_counts1, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list1 = [new_data_list[i] for i, token_count in enumerate(token_counts1) if token_count < percentile_75_1]
    # print(len(new_data_list1))
    # final_list = random.sample(new_data_list1,20000)
    return new_data_list1
def main():

    # file_path_what = '/home/zxw/llm/dataset_base/what_train.jsonl'
    # database_what_all = get_database(file_path_what)
    # database_what = sample(database_what_all)
    # print(len(database_what))
    # with jsonlines.open('/home/zxw/llm/similar_shot/sample_train/what_train_sample.jsonl','w') as f:
        
    #     for dat in database_what:
    #         f.write(dat)
    
    file_path_how = '/home/zxw/llm/dataset_base/how_train.jsonl'
    database_how_all = get_database(file_path_how)
    database_how = sample(database_how_all)
    print(len(database_how))
    with jsonlines.open('/home/zxw/llm/similar_shot/sample_train/how_train_sample.jsonl','w') as f:
        for dat in database_how:
            f.write(dat)

    
    # file_path_why = '/home/zxw/llm/dataset_base/why_train.jsonl'
    # database_why_all = get_database(file_path_why)
    # database_why = sample(database_why_all)
    # print(len(database_why))
    # with jsonlines.open('/home/zxw/llm/similar_shot/sample_train/why_train_sample.jsonl','w') as f:
    #     for dat in database_why:
    #         f.write(dat)

    


if __name__ =='__main__':
    main()


