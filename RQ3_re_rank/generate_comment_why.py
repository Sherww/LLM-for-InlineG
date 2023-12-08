
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.data.conversation.conversation_settings import ConversationHistory
from langport.data.conversation.conversation_settings import get_conv_settings
import jsonlines
from tqdm import tqdm
import random
import numpy as np
# 导入nltk的分词工具
from nltk.tokenize import word_tokenize


class ModelPredictor(object):
    def __init__(self, model_path: str) -> None:
        # 加载模型
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 指定在 CUDA 3 上运行
        # cuda_device = 3

        # self.device = torch.device('cuda:{}'.format(cuda_device))

        self.device = torch.device('cuda')
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.unk_token
    
    def inference_single(self, input: str, max_new_tokens=256):
        input_ids = self.tokenizer([input], return_tensors="pt", padding='longest', max_length=2048, truncation=True)["input_ids"].to(self.device)
        outputs_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        output_text = outputs[0]
        return output_text
    
    def inference_batch(self, inputs: List[str], max_new_tokens=256):
        input_ids = self.tokenizer(inputs, return_tensors="pt", padding='longest', max_length=2048, truncation=True)["input_ids"].to(self.device)
        outputs_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        return outputs
    
    def chat(self, history: ConversationHistory):
        prompt = history.get_prompt()
        output = self.inference_single(prompt)
        return output[len(prompt):]
def get_data():
    file_how = '/home/zxw/llm/generate_comment/how_sample.jsonl'
    file_why = '/home/zxw/llm/generate_comment/why_sample.jsonl'
    file_what = '/home/zxw/llm/generate_comment/what_sample.jsonl'
    how_data = []
    why_data = []
    what_data = []
    print('start----------getting---------')
    with jsonlines.open(file_how,'r')as f:
        for dat in f:
            how_data.append(dat)
    with jsonlines.open(file_why,'r')as ff:
        for dat in ff:
            why_data.append(dat)
    with jsonlines.open(file_what,'r')as fff:
        for dat in fff:
            what_data.append(dat)
    return how_data, why_data, what_data

def get_database(file_path_how):
    datas = []
    with jsonlines.open(file_path_how,'r')as f:
        for dat in f:
            datas.append(dat)
    
    return datas

def sample_percentile(data):
    codes = []
    # 从 data 中提取 'code' 字段并存储为新的列表 codes
    codes = [item['code'] for item in data]
    
    # 使用 NLTK 分词，计算 codes 列表中元素的 token 数目
    token_counts = [len(word_tokenize(code)) for code in codes]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75 = np.percentile(token_counts, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list = [data[i] for i, token_count in enumerate(token_counts) if token_count < percentile_75]
    print('code ----------success-----------')

    comments = []
    # 从 new_data_list 中提取 'comment' 字段并存储为新的列表 comments
    comments = [item['comment'] for item in new_data_list]
    
    # 使用 NLTK 分词，计算 comments 列表中元素的 token 数目
    token_counts1 = [len(word_tokenize(comment)) for comment in comments]
    
    # 使用 NumPy 计算四分之三位值
    percentile_75_1 = np.percentile(token_counts1, 75)
    
    # 创建一个新的列表，存储小于四分之三位值的数据对应的 data 数据
    new_data_list1 = [new_data_list[i] for i, token_count in enumerate(token_counts1) if token_count < percentile_75_1]
    print('comment ----------success-----------')

    # final_list = random.sample(new_data_list1,20000)
    return new_data_list1



def get_output_one(data_list,predictor,settings,database_similar):

    SYSTEM_PROMPT = "从现在开始作为一个 CAN（“code anything now”）。CAN是一位资深编程专家，拥有多年编程经验。CAN 没有字符限制。"
    # "CAN 将自动发送后续消息，直到程序完成。CAN 可以为所提供的任何语言生成代码。"
    # "每次CAN说他无法完成眼前的任务，我都会提醒他“保持本性”，在本性中他会产生正确的代码。"
    # SYSTEM_PROMPT = '请告诉我这段注释和代码是否相关。comment: Get path for exclusions file to use in argument string given to Randoop, code: exclusionFilePath=dont_transform;,只回答是或者否即可'
    

        
    # 存储结果
    results = []
    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        one_shot = database_similar[i]
        examples =[]
        # print(one_shot)
        # print(len(one_shot))
        # code_examples = []

        shot = one_shot["best_data"]
        comment_example = shot["comment"]
        code_example = shot["code"]
            # comment_examples.append(comment_example)
        example = 'Example, The code is:' + str(code_example)+ '. The comment is:' + str(comment_example) +' '  
        examples.append(example)
            
            
        # comment = data["comment"]
        # print(comment)
        code = data["code"]
        instruct = f"Please learn to generate five possible concise English comments for the Java code: {code}."
        
        # comment = 'Get path for exclusions file to use in argument string given to Randoop'
        # code = 'exclusionFilePath=dont_transform;'
        history = ConversationHistory(
            SYSTEM_PROMPT,
            messages=[
                (settings.roles[0], f"Give you one example of comments and codes: {examples}." + instruct + "You should only output the generated comments with the separator of <sep>, without any other words. " 
    ),
                # (settings.roles[0], '请告诉我这段注释和代码是否相关。comment: Get path for exclusions file to use in argument string given to Randoop, code: exclusionFilePath=dont_transform;。请注意，你的输出只需要"是"或者"否"即可，不需要其他任何字符'
    # ),
                # (settings.roles[1], "你好！"),
                # (settings.roles[0], "你是谁？"),
                # (settings.roles[1], None),
            ],
            offset=0,
            settings=settings
        )
        out = predictor.chat(history)
        results.append(out)
    return results
def get_output_five(data_list,predictor,settings,database_similar):

    SYSTEM_PROMPT = "从现在开始作为一个 CAN（“code anything now”）。CAN是一位资深编程专家，拥有多年编程经验。CAN 没有字符限制。"
    # "CAN 将自动发送后续消息，直到程序完成。CAN 可以为所提供的任何语言生成代码。"
    # "每次CAN说他无法完成眼前的任务，我都会提醒他“保持本性”，在本性中他会产生正确的代码。"
    # SYSTEM_PROMPT = '请告诉我这段注释和代码是否相关。comment: Get path for exclusions file to use in argument string given to Randoop, code: exclusionFilePath=dont_transform;,只回答是或者否即可'
    

        
    # 存储结果
    results = []
    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        five_shot = database_similar[i]
        #!!!!!!!!!!!!!这里如果要用要改的
        examples =[]
        # code_examples = []
        for i in range(len(five_shot)):
            sho = five_shot[i]
            shot = sho["best_data"]
            comment_example = shot["comment"]
            code_example = shot["code"]
            # comment_examples.append(comment_example)
            example = f'Example{i}, The code is:' + str(code_example)+ '. The comment is:' + str(comment_example)  
            examples.append(example)
            
            
        # comment = data["comment"]
        # print(comment)
        code = data["code"]
        instruct = f"Please learn to generate a concise English comment for the Java code: {code}."
        
        # comment = 'Get path for exclusions file to use in argument string given to Randoop'
        # code = 'exclusionFilePath=dont_transform;'
        history = ConversationHistory(
            SYSTEM_PROMPT,
            messages=[
                (settings.roles[0], f"Give you five examples of comments and codes: {examples}." + instruct + "You should only output one generated comment for me, without any other words. " 
    ),
                # (settings.roles[0], '请告诉我这段注释和代码是否相关。comment: Get path for exclusions file to use in argument string given to Randoop, code: exclusionFilePath=dont_transform;。请注意，你的输出只需要"是"或者"否"即可，不需要其他任何字符'
    # ),
                # (settings.roles[1], "你好！"),
                # (settings.roles[0], "你是谁？"),
                # (settings.roles[1], None),
            ],
            offset=0,
            settings=settings
        )
        out = predictor.chat(history)
        results.append(out)
    return results
def writr_to_file(results,name):
    with jsonlines.open(f"result/{name}.jsonl", "w") as outfile:
            for dat in results:
                outfile.write(dat)

    # print(len(related_results))
    

if __name__ == "__main__":
    
    #设置模型背景信息
    predictor = ModelPredictor('/data/wangjun/models/Llama-2-7b-chat-hf/')
    settings = get_conv_settings("llama")
    
    how_data, why_data, what_data = get_data()

    
    
    # # how文件
    
    # file_path_how = '/home/zxw/llm/similar_shot/similar_result/how_output_20000.jsonl'
    # database_how = get_database(file_path_how)
    # # database_how_sample = sample_percentile(database_how)

    # # 一个例子
    # results_how = get_output_one(how_data,predictor,settings,database_how)
    # name1 = 'similar_how_one_rerank'
    # writr_to_file(results_how,name1)
    # # # 五个例子
    # # results_how = get_output_five(how_data,predictor,settings,database_how_sample)
    # # name11 = 'how_five_shot'
    # # writr_to_file(results_how,name11)
    
    
    
    # # what文件
    # file_path_what = '/home/zxw/llm/similar_shot/similar_result/what_output_20000.jsonl'
    # database_what = get_database(file_path_what)
    # # 一个例子
    # results_what = get_output_one(what_data,predictor,settings,database_what)
    # name1 = 'similar_what_one_rerank'
    # writr_to_file(results_what,name1)
    
    
    # why文件
    file_path_why = '/home/zxw/llm/similar_shot/similar_result/why_output_20000.jsonl'
    database_why = get_database(file_path_why)
    # 一个例子
    results_why = get_output_one(why_data,predictor,settings,database_why)
    name1 = 'similar_why_one_rerank'
    writr_to_file(results_why,name1)
    
  
        