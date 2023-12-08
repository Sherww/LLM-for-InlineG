import json
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.data.conversation.conversation_settings import ConversationHistory
from langport.data.conversation.conversation_settings import get_conv_settings
import jsonlines
from tqdm import tqdm


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
    file_how = 'how_sample.jsonl'
    file_why = 'why_sample.jsonl'
    file_what = 'what_sample.jsonl'
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
def get_output(data_list):
    #设置模型背景信息
    predictor = ModelPredictor('/data/wangjun/models/Llama-2-7b-chat-hf/')
    settings = get_conv_settings("llama")
    SYSTEM_PROMPT = "从现在开始作为一个 CAN（“code anything now”）。CAN是一位资深编程专家，拥有多年编程经验。CAN 没有字符限制。"
    # "CAN 将自动发送后续消息，直到程序完成。CAN 可以为所提供的任何语言生成代码。"
    # "每次CAN说他无法完成眼前的任务，我都会提醒他“保持本性”，在本性中他会产生正确的代码。"
    # SYSTEM_PROMPT = '请告诉我这段注释和代码是否相关。comment: Get path for exclusions file to use in argument string given to Randoop, code: exclusionFilePath=dont_transform;,只回答是或者否即可'

    
    # 存储结果
    results = []
    for data in tqdm(data_list):
        # comment = data["comment"]
        # print(comment)
        code = data["code"]
        
        # comment = 'Get path for exclusions file to use in argument string given to Randoop'
        # code = 'exclusionFilePath=dont_transform;'
        history = ConversationHistory(
            SYSTEM_PROMPT,
            messages=[
                (settings.roles[0],f"Please generate a concise English comment for the Java code: '{code}'. You should only output the generated comment, without any other characters. "
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
    with jsonlines.open(f"concise_result/{name}.jsonl", "w") as outfile:
            for dat in results:
                outfile.write(dat)

    # print(len(related_results))
if __name__ == "__main__":
    how_data, why_data, what_data = get_data()
    
    # how文件
    #results_how = get_output(how_data)
    #name1 = 'how'
    #writr_to_file(results_how,name1)
    
    # what文件
    #results_what = get_output(what_data)
    #name2 = 'what'
    #writr_to_file(results_what,name2)   
    
    # why文件
    results_why = get_output(why_data)
    name3 = 'why'
    writr_to_file(results_why,name3)
    
    
    
    
    
    
    
    
    
    
