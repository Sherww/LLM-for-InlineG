# import jsonlines
# from nltk.translate.bleu_score import corpus_bleu
# from rouge import Rouge
# from meteor import meteor_score

# # 读取数据
# data = []
# with jsonlines.open('why_cg_bart_test_out-20230310-082748.jsonl') as f:
#     for line in f:
#         data.append(line)

# # 提取参考文本和目标文本
# references = []
# targets = []
# for example in data:
#     references.append([example['comment']])
#     targets.append(example['prediction'])

# # 计算BLEU指标
# weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
# bleu_scores = []
# for i in range(4):
#     bleu_scores.append(corpus_bleu(references, targets, weights=weights[i]))

# # 计算ROUGE-L指标
# rouge_scores = Rouge().get_scores(targets, references, avg=True)

# # 计算METEOR指标
# meteor_scores = meteor_score(targets, references)

# # 输出结果
# print('BLEU-1:', bleu_scores[0])
# print('BLEU-2:', bleu_scores[1])
# print('BLEU-3:', bleu_scores[2])
# print('BLEU-4:', bleu_scores[3])
# print('ROUGE-L:', rouge_scores['rouge-l']['f'])
# print('METEOR:', meteor_scores)


import jsonlines
from nltk.translate import meteor_score
from rouge import Rouge
from rouge_score import rouge_scorer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction
import re

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score

smoother = SmoothingFunction()


def random_one_result(name):
    # 打开 JSONL 文件并读取数据
    data_cans = []
    # data_cans2 = []
    data_refs = []
    data_refs2 = []

    # with jsonlines.open('/home/zxw/llm/random_shot/result/how_one_shot.jsonl') as f:
    # with jsonlines.open('/home/zxw/llm/random_shot/result/what_one_shot.jsonl') as f:
    with jsonlines.open(f'/home/zxw/llm/random_shot/result/{name}_one_shot.jsonl') as f:

        for line in f:
            line = line.split('\n\n')[-1]
            line = line.replace('\n','')
            line =  re.sub(r'[^\w\s]', '', line)
            data_cans.append(line)
    # print(data_cans[0:10])
            # data_cans2.append(line)
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    # 初始化 ROUGE 和 SmoothingFunction 对象
    rouge = Rouge()
    # smoothing_function = SmoothingFunction().method1
    
    # 初始化 BLEU 和 Meteor 计算的变量
    # bleu1_scores = []
    # bleu2_scores = []
    # bleu3_scores = []
    # bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    
    # 计算BLEU指标
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    # for example in data:
    #     nee=example.split(',')
    #     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    #     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
    
    for i in range(4):
        bleu_scores.append(corpus_bleu(data_refs2, data_cans, weights=weights[i],smoothing_function=smoother.method1))
    
    # 迭代 JSONL 文件中的每一行
    for i in tqdm(range(len(data_refs))):
        
            reference = data_refs[i]
            target = data_cans[i]
            
            # target = ''.join(target)
            reference_tokens = reference.split()
            target_tokens = target.split()
            
        
            # 计算 ROUGE-L 值
            try:
                        
                # !!!!!!!!!!!!! 计算 ROUGE-L 值
                # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
                
                # Calculate ROUGE scores
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_score = scorer.score(target, reference)['rougeL'].fmeasure
                
                # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
            except:
                print(target +'\n')
                print(reference)
            # 添加 ROUGE-L 值到列表中
            rouge_scores.append(rouge_score)
            
            # 计算 Meteor 值
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            
            # 添加 Meteor 值到列表中
            meteor_scores.append(meteor_scoree)

    
    # 打印平均值
    print('BLEU-1:', bleu_scores[0]*100)
    print('BLEU-2:', bleu_scores[1]*100)
    print('BLEU-3:', bleu_scores[2]*100)
    print('BLEU-4:', bleu_scores[3]*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
    print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)
def random_five_result(name):
    # 打开 JSONL 文件并读取数据
    data_cans = []
    # data_cans2 = []
    data_refs = []
    data_refs2 = []

    # with jsonlines.open('/home/zxw/llm/random_shot/result/how_five_shot.jsonl') as f:
    # with jsonlines.open('/home/zxw/llm/random_shot/result/what_five_shot.jsonl') as f:
    with jsonlines.open(f'/home/zxw/llm/random_shot/result/{name}_five_shot.jsonl') as f:

        for lin in f:
            line = lin.split('\n\n')[-1]
            line = line.replace('\n','')
            line =  re.sub(r'[^\w\s]', '', line)
            if line == None:
                print(lin)
                line = 'None_result'
                
            data_cans.append(line)
    # print(data_cans[0:10])
            # data_cans2.append(line)
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    # 初始化 ROUGE 和 SmoothingFunction 对象
    rouge = Rouge()
    # smoothing_function = SmoothingFunction().method1
    
    # 初始化 BLEU 和 Meteor 计算的变量
    # bleu1_scores = []
    # bleu2_scores = []
    # bleu3_scores = []
    # bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    
    # 计算BLEU指标
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    # for example in data:
    #     nee=example.split(',')
    #     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    #     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
    
    for i in range(4):
        bleu_scores.append(corpus_bleu(data_refs2, data_cans, weights=weights[i],smoothing_function=smoother.method1))
    
    # 迭代 JSONL 文件中的每一行
    for i in tqdm(range(len(data_refs))):
        
            reference = data_refs[i]
            target = data_cans[i]
            
            # target = ''.join(target)
            reference_tokens = reference.split()
            target_tokens = target.split()
            
        
            # 计算 ROUGE-L 值
            try:
                        
                # !!!!!!!!!!!!! 计算 ROUGE-L 值
                # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
                
                # Calculate ROUGE scores
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_score = scorer.score(target, reference)['rougeL'].fmeasure
            
            except:
                print(target +'\n')
                print(reference)
            # 添加 ROUGE-L 值到列表中
            rouge_scores.append(rouge_score)
            
            # 计算 Meteor 值
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            
            # 添加 Meteor 值到列表中
            meteor_scores.append(meteor_scoree)

    
    # 打印平均值
    print('BLEU-1:', bleu_scores[0]*100)
    print('BLEU-2:', bleu_scores[1]*100)
    print('BLEU-3:', bleu_scores[2]*100)
    print('BLEU-4:', bleu_scores[3]*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
    print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)

def concise_result(name):
    # 打开 JSONL 文件并读取数据
    data_cans = []
    # data_cans2 = []
    data_refs = []
    data_refs2 = []

    # with jsonlines.open('/home/zxw/llm/generate_comment/concise_result/how.jsonl') as f:
    # with jsonlines.open('/home/zxw/llm/generate_comment/concise_result/what.jsonl') as f:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/concise_result/{name}.jsonl') as f:

        for line in f:
            line = line.replace('\n','')
            line =  re.sub(r'[^\w\s]', '', line)
            data_cans.append(line)
            # data_cans2.append(line)
            
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    # 初始化 ROUGE 和 SmoothingFunction 对象
    rouge = Rouge()
    # smoothing_function = SmoothingFunction().method1
    
    # 初始化 BLEU 和 Meteor 计算的变量
    # bleu1_scores = []
    # bleu2_scores = []
    # bleu3_scores = []
    # bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    
    # 计算BLEU指标
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    # for example in data:
    #     nee=example.split(',')
    #     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    #     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
    
    for i in range(4):
        bleu_scores.append(corpus_bleu(data_refs2, data_cans, weights=weights[i],smoothing_function=smoother.method1))
    
    # 迭代 JSONL 文件中的每一行
    for i in tqdm(range(len(data_refs))):
        
            reference = data_refs[i]
            target = data_cans[i]
            
            # target = ''.join(target)
            reference_tokens = reference.split()
            target_tokens = target.split()
            
        
        
            # !!!!!!!!!!!!! 计算 ROUGE-L 值
            # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(target, reference)['rougeL'].fmeasure
            
            
            # 添加 ROUGE-L 值到列表中
            rouge_scores.append(rouge_score)
            
            # 计算 Meteor 值
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            
            # 添加 Meteor 值到列表中
            meteor_scores.append(meteor_scoree)

    
    # 打印平均值
    print('BLEU-1:', bleu_scores[0]*100)
    print('BLEU-2:', bleu_scores[1]*100)
    print('BLEU-3:', bleu_scores[2]*100)
    print('BLEU-4:', bleu_scores[3]*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
    print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)
def codebert_result(name):
    # 打开 JSONL 文件并读取数据
    data_cans = []
    # data_cans2 = []
    data_refs = []
    data_refs2 = []

    # with open('/home/zxw/llm/codebert/code/model/how/how_test_1_4.output') as f:
    # with open('/home/zxw/llm/codebert/code/model/what/what_test_1_4.output') as f:
    with open(f'/home/zxw/llm/codebert/code/model/{name}/{name}_test_1_4.output') as f:

        datas = f.readlines()
        for line in datas:
            line = re.sub(r'^\d+\s*', '', line)
            line = line.replace('\n','')
            line =  re.sub(r'[^\w\s]', '', line)
            data_cans.append(line)
            # data_cans2.append(line)
            
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    # 初始化 ROUGE 和 SmoothingFunction 对象
    rouge = Rouge()
    # smoothing_function = SmoothingFunction().method1
    
    # 初始化 BLEU 和 Meteor 计算的变量
    # bleu1_scores = []
    # bleu2_scores = []
    # bleu3_scores = []
    # bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    
    # 计算BLEU指标
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    # for example in data:
    #     nee=example.split(',')
    #     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    #     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
    
    for i in range(4):
        bleu_scores.append(corpus_bleu(data_refs2, data_cans, weights=weights[i],smoothing_function=smoother.method1))
        
    # 迭代 JSONL 文件中的每一行
    for i in tqdm(range(len(data_refs))):
        
            reference = data_refs[i]
            target = data_cans[i]
            
            # target = ''.join(target)
            reference_tokens = reference.split()
            target_tokens = target.split()
            
        
        
            # !!!!!!!!!!!!! 计算 ROUGE-L 值
            # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(target, reference)['rougeL'].fmeasure
            
            
            # 添加 ROUGE-L 值到列表中
            rouge_scores.append(rouge_score)
            
            # 计算 Meteor 值
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            
            # 添加 Meteor 值到列表中
            meteor_scores.append(meteor_scoree)

    
    # 打印平均值
    print('BLEU-1:', bleu_scores[0]*100)
    print('BLEU-2:', bleu_scores[1]*100)
    print('BLEU-3:', bleu_scores[2]*100)
    print('BLEU-4:', bleu_scores[3]*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
    print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)
def similar_one_result(name):
    # 打开 JSONL 文件并读取数据
    data_cans = []
    # data_cans2 = []
    data_refs = []
    data_refs2 = []

    # with open('/home/zxw/llm/codebert/code/model/how/how_test_1_4.output') as f:
    # with open('/home/zxw/llm/codebert/code/model/what/what_test_1_4.output') as f:
    with jsonlines.open(f'/home/zxw/llm/similar_shot/result/similar_{name}_one_shot.jsonl') as f:

        for lin in f:
            line = lin.split('\n\n')[-1]
            line = line.replace('\n','')
            line =  re.sub(r'[^\w\s]', '', line)
            if line == None:
                print(lin)
                line = 'None_result'
                
            data_cans.append(line)
            # data_cans2.append(line)
            
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    # 初始化 ROUGE 和 SmoothingFunction 对象
    rouge = Rouge()
    # smoothing_function = SmoothingFunction().method1
    
    # 初始化 BLEU 和 Meteor 计算的变量
    # bleu1_scores = []
    # bleu2_scores = []
    # bleu3_scores = []
    # bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    
    # 计算BLEU指标
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    # for example in data:
    #     nee=example.split(',')
    #     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    #     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
    
    for i in range(4):
        bleu_scores.append(corpus_bleu(data_refs2, data_cans, weights=weights[i],smoothing_function=smoother.method1))
    # print(bleu_scores)
    
    # 迭代 JSONL 文件中的每一行
    for i in tqdm(range(len(data_refs))):
        
            reference = data_refs[i]
            target = data_cans[i]
            
            # target = ''.join(target)
            reference_tokens = reference.split()
            target_tokens = target.split()
            
        
            # !!!!!!!!!!!!! 计算 ROUGE-L 值
            # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(target, reference)['rougeL'].fmeasure
            
            # 添加 ROUGE-L 值到列表中
            rouge_scores.append(rouge_score)
            
            # 计算 Meteor 值
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            
            # 添加 Meteor 值到列表中
            meteor_scores.append(meteor_scoree)

    
    # 打印平均值
    print('BLEU-1:', bleu_scores[0]*100)
    print('BLEU-2:', bleu_scores[1]*100)
    print('BLEU-3:', bleu_scores[2]*100)
    print('BLEU-4:', bleu_scores[3]*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
    print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)
def rerank(name):
    smoother = SmoothingFunction()

    # 打开 JSONL 文件并读取数据
    data_cans1 = []
    data_cans2 = []
    data_cans3 = []
    data_cans4 = []
    data_cans5 = []
    
    # data_cans2 = []
    data_refs = []
    data_refs2 = []
    datas = []
    # with open('/home/zxw/llm/codebert/code/model/how/how_test_1_4.output') as f:
    # with open('/home/zxw/llm/codebert/code/model/what/what_test_1_4.output') as f:
    with jsonlines.open(f'/home/zxw/llm/re_rank/result/similar_{name}_one_rerank.jsonl') as f:
        for da in f:
            datas.append(da)
        for i in tqdm(range(len(datas))):
            lin = datas[i]
            line = lin.split('\n<sep>')
            linee = lin.split('<sep>')
            lineee = lin.split('</sep>\n\n')[-1]
            lineee = lineee.split('\n')
            try:
                line1 = line[1].replace('</sep>','')
                line1 =  re.sub(r'[^\w\s]', '', line1)
            except:
                try:
                    line1 = linee[1].replace('</sep>','').replace('1.','')
                    line1 =  re.sub(r'[^\w\s]', '', line1)
                except:
                    try:
                        line1 = lineee[0].replace('</sep>','').replace('1.','')
                        line1 =  re.sub(r'[^\w\s]', '', line1)
                    except:
                        # print(line)
                        # print(i)

                        line1 = lin.split('\n')[1].replace('</sep>','').replace('1.','')
                        line1 =  re.sub(r'[^\w\s]', '', line1)

            try:
                line2 = line[2].replace('</sep>','')
                line2 =  re.sub(r'[^\w\s]', '', line2)
            except:
                try:
                    line2 = linee[2].replace('</sep>','').replace('2.','')
                    line2 =  re.sub(r'[^\w\s]', '', line2)
                except:
                    try:
                        line2 = lineee[1].replace('</sep>','').replace('2.','')
                        line2 =  re.sub(r'[^\w\s]', '', line2)
                    except:
                        # print(line)
                        # print(i)
                        try:
                            line2 = lin.split('\n')[1].replace('</sep>','').replace('2.','')
                            line2 =  re.sub(r'[^\w\s]', '', line1)
                        except:
                            line2 = re.sub(r'[^\w\s]', '', lin)
            try:
                line3 = line[3].replace('</sep>','')
                line3 =  re.sub(r'[^\w\s]', '', line3)
            except:
                try:
                    line3 = linee[3].replace('</sep>','').replace('3.','')
                    line3 =  re.sub(r'[^\w\s]', '', line3)
                except:
                    try:
                        line3 = lineee[2].replace('</sep>','').replace('3.','')
                        line3 =  re.sub(r'[^\w\s]', '', line3)
                    except:
                        # print(line)
                        # print(i)
                        # line3 = 'None result'
                        try:
                            line3 = lin.split('\n')[1].replace('</sep>','').replace('3.','')
                            line3 =  re.sub(r'[^\w\s]', '', line1)
                        except:
                            line3 = re.sub(r'[^\w\s]', '', lin)
            try:
                line4 = line[4].replace('</sep>','')
                line4 =  re.sub(r'[^\w\s]', '', line4)
            except:
                try:
                    line4 = linee[4].replace('</sep>','').replace('4.','')
                    line4 =  re.sub(r'[^\w\s]', '', line4)
                except:
                    try:
                        line4 = lineee[3].replace('</sep>','').replace('4.','')
                        line4 =  re.sub(r'[^\w\s]', '', line4)
                    except:
                        # print(line)
                        # print(i)
                        # line4 = 'None result'
                        try:
                            line4 = lin.split('\n')[1].replace('</sep>','').replace('4.','')
                            line4 =  re.sub(r'[^\w\s]', '', line1)
                        except:
                            line4 = re.sub(r'[^\w\s]', '', lin)
            try:
                line5 = line[5].replace('</sep>','')
                line5 =  re.sub(r'[^\w\s]', '', line5)
            except:
                try:
                    line5 = linee[5].replace('</sep>','').replace('5.','')
                    line5 =  re.sub(r'[^\w\s]', '', line5)
                except:
                    try:
                        line5 = lineee[4].replace('</sep>','').replace('5.','')
                        line5 =  re.sub(r'[^\w\s]', '', line5)
                    except:
                        # print(line)
                        # print(i)
                        # line5 = 'None result'
                        try:
                            line5 = lin.split('\n')[1].replace('</sep>','').replace('5.','')
                            line5 =  re.sub(r'[^\w\s]', '', line1)
                        except:
                            line5 = re.sub(r'[^\w\s]', '', lin)
            # line = line.replace('\n','')
            # line =  re.sub(r'[^\w\s]', '', line)
            # if line1 == None:
            #     print(lin)
            #     line1 = 'None_result'
            # if line2 == None:
            #     print(lin)
            #     line2 = 'None_result'                
            # if line3 == None:
            #     print(lin)
            #     line3 = 'None_result'
            # if line4 == None:
            #     print(lin)
            #     line4 = 'None_result'
            # if line5 == None:
            #     print(lin)
            #     line5 = 'None_result'


            data_cans1.append(line1)
            data_cans2.append(line2)
            data_cans3.append(line3)
            data_cans4.append(line4)
            data_cans5.append(line5)
            
    # print(data_cans5[0:10])        
    cans = [data_cans1,data_cans2,data_cans3,data_cans4,data_cans5]        
                        
    # with jsonlines.open('/home/zxw/llm/generate_comment/how_sample.jsonl') as ff:
    # with jsonlines.open('/home/zxw/llm/generate_comment/what_sample.jsonl') as ff:
    with jsonlines.open(f'/home/zxw/llm/generate_comment/{name}_sample.jsonl') as ff:

        for line in ff:
            comment = line['comment']
            comment = comment.replace('\n','')
            comment =  re.sub(r'[^\w\s]', '', comment)
            data_refs.append(comment)
            data_refs2.append([comment])


    best_bleu_scores = []
    best_rouge_scores = []
    best_meteor_scores = []



    for i in tqdm(range(len(data_refs))):
        bleu_score_all = []
        bleu_scores_4 = []
        bleu_scores_3 = []
        bleu_scores_2 = []
        bleu_scores_1 = []
        rouge_scores = []
        meteor_scores = []
        ref = data_refs[i]

        for data_cans in cans:
            bleu_scores=[]

            candidate = data_cans[i]
            reference_tokens = ref.split()
            target_tokens = candidate.split()
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(candidate, ref)['rougeL'].fmeasure
            rouge_scores.append(rouge_score)
            
            # rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']

            # Calculate METEOR scores
            meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
            meteor_scores.append(meteor_scoree)
            
            
            weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
            for j in range(4):
                # Calculate BLEU scores
                bleu_score = corpus_bleu([[ref]], [candidate], weights=weights[j],smoothing_function=smoother.method1)
                # print(bleu_score)
                bleu_scores.append(bleu_score)
                
            bleu_score_all.append(bleu_scores)
            # print(bleu_score_all)
        for k in range(len(bleu_score_all)):
                
                bleu_scores_4.append(bleu_score_all[k][3])
                bleu_scores_3.append(bleu_score_all[k][2])
                bleu_scores_2.append(bleu_score_all[k][1])
                bleu_scores_1.append(bleu_score_all[k][0])

        # print(bleu_scores_4)

        # 这样可以返回最优结果对应的comment文本是什么
        # best_bleu_index = bleu_scores.index(max(bleu_scores))
        # best_rouge_index = rouge_scores.index(max(rouge_scores))
        # best_meteor_index = meteor_scores.index(max(meteor_scores))

        # best_bleu_scores.append(cans[best_bleu_index][i])
        # best_rouge_scores.append(cans[best_rouge_index][i])
        # best_meteor_scores.append(cans[best_meteor_index][i])
  
    
        best_bleu_index = bleu_scores_4.index(max(bleu_scores_4))
        # print(best_bleu_index)
        # best_bleu= max(bleu_scores_4)
        best_rouge = max(rouge_scores)
        best_meteor = max(meteor_scores)

        best_bleu_scores.append(bleu_score_all[best_bleu_index])
        best_rouge_scores.append(best_rouge)
        best_meteor_scores.append(best_meteor)
    # return best_bleu_scores, best_rouge_scores, best_meteor_scores

# # 示例用法
# refs = ["reference1", "reference2", "reference3"]
# cans1 = ["candidate1_1", "candidate2_1", "candidate3_1"]
# cans2 = ["candidate1_2", "candidate2_2", "candidate3_2"]
# cans3 = ["candidate1_3", "candidate2_3", "candidate3_3"]
# cans4 = ["candidate1_4", "candidate2_4", "candidate3_4"]
# cans5 = ["candidate1_5", "candidate2_5", "candidate3_5"]

# cans = [cans1, cans2, cans3, cans4, cans5]

# best_bleu, best_rouge, best_meteor = calculate_scores(refs, cans)

# print("Best BLEU Scores:", best_bleu)
# print("Best ROUGE Scores:", best_rouge)
# print("Best METEOR Scores:", best_meteor)


    best_bleu_scores_0 = []
    best_bleu_scores_1 = []
    best_bleu_scores_2 = []
    best_bleu_scores_3 = []
    # 打印平均值
    # print('BLEU-1:', best_bleu_scores[0]*100)
    # print('BLEU-2:', best_bleu_scores[1]*100)
    # print('BLEU-3:', best_bleu_scores[2]*100)
    # print('BLEU-4:', best_bleu_scores[3]*100)
    for l in tqdm(range(len(best_bleu_scores))):
        best_bleu_scores_0.append(best_bleu_scores[l][0])
        best_bleu_scores_1.append(best_bleu_scores[l][1])
        best_bleu_scores_2.append(best_bleu_scores[l][2])
        best_bleu_scores_3.append(best_bleu_scores[l][3])
    print('BLEU-1:', (sum(best_bleu_scores_0) / len(best_bleu_scores_0))*100)
    print('BLEU-2:', (sum(best_bleu_scores_1) / len(best_bleu_scores_1))*100)
    print('BLEU-3:', (sum(best_bleu_scores_2) / len(best_bleu_scores_2))*100)
    print('BLEU-4:', (sum(best_bleu_scores_3) / len(best_bleu_scores_3))*100)
    # print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
    # print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
    # print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
    # print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
    # print(best_rouge_scores)
    print('ROUGE-L:', (sum(best_rouge_scores) / len(best_rouge_scores))*100)
    print('Meteor:', (sum(best_meteor_scores) / len(best_meteor_scores))*100)
    
if __name__ =='__main__':
    
    # name = 'why'
    # name = 'what'
    name = 'how'
    
    print(f'start-----------{name}------codebert_result')
    codebert_result(name)
    
    
    print(f'start-----------{name}------concise_result')    
    concise_result(name)
    
    
    print(f'start-----------{name}------random_one_result')
    random_one_result(name)
    
    
    print(f'start-----------{name}------random_five_result')
    random_five_result(name)
    
    
    print(f'start-----------{name}------similar_one_result')
    similar_one_result(name)
    
    print(f'start-----------{name}------rerank')
    rerank(name)