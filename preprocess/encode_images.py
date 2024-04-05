import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
# # CLIPTokenizer,
# # CLIPTextModelWithProjection,
CLIPVisionModelWithProjection,
CLIPImageProcessor,
ChineseCLIPModel,
ChineseCLIPProcessor
)
from PIL import Image

# clip_id = "openai/clip-vit-base-patch32"
# tokenizer = CLIPTokenizer.from_pretrained(clip_id)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id)
# image_processor = CLIPImageProcessor.from_pretrained(clip_id)
# image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id)

# model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
# processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# # compute image feature
# inputs = processor(images=image, return_tensors="pt")   # 先通过processor将图片转换为tensor
# image_features = model.get_image_features(**inputs)     # 再通过model将图片转换为特征
# image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#
# # compute text feature
# inputs = processor(text=text, padding=True, return_tensors="pt")    # 先通过processor将文本转换为tensor
# text_features = model.get_text_features(**inputs)                   # 再通过model将文本转换为特征
# text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#
# # compute image-text similarity scores
# # inputs = processor(text=text, images=image, padding=True, return_tensors="pt")
# # outputs = model(**inputs)
# # logits_per_image = outputs.logits_per_image # # this is the image-text similarity score
# # probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
#
#
# # logits_per_image = image_features @ text_features.t()
# logits_per_image = torch.mm(image_features, text_features.t())
#


# clip_id = "openai/clip-vit-base-patch32"
clip_id = "OFA-Sys/chinese-clip-vit-base-patch16"
# tokenizer = CLIPTokenizer.from_pretrained(clip_id)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id)
# image_processor = CLIPImageProcessor.from_pretrained(clip_id)
# image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id)
image_processor = ChineseCLIPProcessor.from_pretrained(clip_id)
image_encoder = ChineseCLIPModel.from_pretrained(clip_id)

device = "cuda" if torch.cuda.is_available() else "cpu"

image_encoder.to(device)

# 读取照片，提取照片特征，并保存为txt
# model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
# processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
# 照片编号为0-10724共10725张照片
images_lp_path = '/mnt/nvme0n1/VisCPM-main/results/big/long_prompt'
# 读取照片
images_lp = []
for i in range(10725):
    images_lp.append(f'{images_lp_path}/'+str(i)+'.jpg')
# 提取照片特征
image_lp_features = []
with torch.no_grad():
    for image in tqdm(images_lp):
        image = Image.open(image).convert("RGB")  # 读取图片
        preprocessed_image = image_processor(images=image, return_tensors="pt").to(device)  # 先通过processor将图片转换为tensor
        # image_feature = image_encoder(**preprocessed_image).image_embeds     # 再通过model将图片转换为特征
        image_feature = image_encoder.get_image_features(**preprocessed_image)     # 再通过model将图片转换为特征
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True) # 先不归一化
        image_lp_features.append(image_feature)
# 转化为numpy数组
image_lp_features = [feature.cpu() for feature in image_lp_features]
image_lp_features = torch.cat(image_lp_features, dim=0).numpy()
# 写进txt文件
print('image_big_lp_features:', image_lp_features.shape)
np.savetxt('image_big_lp_features.txt', image_lp_features, fmt='%.15e')

images_sp_path = '/mnt/nvme0n1/VisCPM-main/results/big/short_prompt'
images_sp = []
for i in range(10725):
    images_sp.append(f'{images_sp_path}/'+str(i)+'.jpg')
image_sp_features = []
with torch.no_grad():
    for image in tqdm(images_sp):
        image=Image.open(image).convert("RGB")
        preprocessed_image = image_processor(images=image, return_tensors="pt").to(device)  # 先通过processor将图片转换为tensor
        # image_feature = image_encoder(**preprocessed_image).image_embeds     # 再通过model将图片转换为特征
        image_feature = image_encoder.get_image_features(**preprocessed_image)     # 再通过model将图片转换为特征
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True) # 先不归一化
        image_sp_features.append(image_feature)
image_sp_features = [feature.cpu() for feature in image_sp_features]
image_sp_features = torch.cat(image_sp_features, dim=0).numpy()
print('image_big_sp_features:', image_sp_features.shape)
np.savetxt('image_big_sp_features.txt', image_sp_features)

# 获取small的ids
small_ids = []
with open('new_videos_small.txt', 'r') as f:
    for line in f.readlines():
        small_ids.append(int(line.strip()))
image_small_lp = []
image_small_sp = []
for i in small_ids:
    image_small_lp.append(f'{images_lp_path}/'+str(i)+'.jpg')
    image_small_sp.append(f'{images_sp_path}/'+str(i)+'.jpg')
image_small_lp_features = []
image_small_sp_features = []
with torch.no_grad():
    for image in tqdm(image_small_lp):
        image=Image.open(image).convert("RGB")
        preprocessed_image = image_processor(images=image, return_tensors="pt").to(device)  # 先通过processor将图片转换为tensor
        # image_feature = image_encoder(**preprocessed_image).image_embeds     # 再通过model将图片转换为特征
        image_feature = image_encoder.get_image_features(**preprocessed_image)     # 再通过model将图片转换为特征
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True) # 先不归一化
        image_small_lp_features.append(image_feature)
    for image in tqdm(image_small_sp):
        image=Image.open(image).convert("RGB")
        preprocessed_image = image_processor(images=image, return_tensors="pt").to(device)  # 先通过processor将图片转换为tensor
        # image_feature = image_encoder(**preprocessed_image).image_embeds     # 再通过model将图片转换为特征
        image_feature = image_encoder.get_image_features(**preprocessed_image)     # 再通过model将图片转换为特征
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True) # 先不归一化
        image_small_sp_features.append(image_feature)

image_small_lp_features = [feature.cpu() for feature in image_small_lp_features]
image_small_sp_features = [feature.cpu() for feature in image_small_sp_features]
image_small_lp_features = torch.cat(image_small_lp_features, dim=0).numpy()
image_small_sp_features = torch.cat(image_small_sp_features, dim=0).numpy()
print('image_small_lp_features:', image_small_lp_features.shape)
print('image_small_sp_features:',image_small_sp_features.shape)

np.savetxt('image_small_lp_features.txt', image_small_lp_features)
np.savetxt('image_small_sp_features.txt', image_small_sp_features)