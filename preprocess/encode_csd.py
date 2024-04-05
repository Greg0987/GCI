
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import numpy as np
import torch
import torch.nn.functional as F

# tokenizer = AutoTokenizer.from_pretrained("hfl/Chinese-bert-wwm")
# model = AutoModel.from_pretrained("hfl/Chinese-bert-wwm")
clip_id = "OFA-Sys/chinese-clip-vit-base-patch16"
tokenizer = ChineseCLIPProcessor.from_pretrained(clip_id)
model = ChineseCLIPModel.from_pretrained(clip_id)

# 读取csv文件
df = pd.read_csv('id2class.csv')
# 读取类别
class_list = df['class'].tolist()
print(class_list)
print(len(class_list))

csd = []
csd_norm = []
# 将类别输入进bert-Chinese中，获取csd向量
# inputs = tokenizer(i, return_tensors='pt')
# outputs = model(**inputs)
inputs = tokenizer(text=class_list, padding=True, return_tensors='pt')
outputs = model.get_text_features(**inputs)
print(outputs.shape)

# last_hidden_states = outputs.last_hidden_state
# pooler_output = outputs.pooler_output
# # 归一化
# pooler_output = F.normalize(pooler_output, p=2, dim=1)
csd.append(outputs)
pooler_output = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
csd_norm.append(pooler_output)
# 用pooler_output作为csd向量
# 将CSD文件保存为txt文件
csd = torch.cat(csd, dim=0)
csd_norm = torch.cat(csd_norm, dim=0)
np.savetxt('csd.txt', csd.detach().numpy())
np.savetxt('csd_norm.txt', csd_norm.detach().numpy())