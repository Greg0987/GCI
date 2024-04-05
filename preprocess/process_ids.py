import os
import json
import numpy as np
import torch
import pandas as pd

# 读取csv，展示数据
videos_path = 'video2tags.csv'
videos = pd.read_csv(videos_path)
v_ids = videos['video_id'].values
v_features = videos['video_tags_name'].values
# 对v_features每个元素进行处理，去掉首尾的中括号，去掉空格，然后按照逗号分割，再去除每个元素的首尾引号
v_features = [v[1:-1].replace(' ', '').split(',') for v in v_features]
v_features = [[v[1:-1] for v in v_feature] for v_feature in v_features]
print(v_features[4])
print(type(v_features[4]))

# 全命令，格式为：'一个短视频，关键词为：xxx、xxx、xxx‘
prompts_long = []
for i in range(len(v_features)):
    prompts_long.append('一个短视频，关键词为：'+'、'.join(v_features[i]))

# 短命令，格式为：'xxx、xxx、xxx‘
prompts_short = []
for i in range(len(v_features)):
    prompts_short.append('、'.join(v_features[i]))

# 写出txt文件
with open('prompts_long_big.txt', 'w') as f:
    for prompt in prompts_long:
        f.write(prompt+'\n')
with open('prompts_short_big.txt', 'w') as f:
    for prompt in prompts_short:
        f.write(prompt+'\n')

# 获取视频重新编号与原id的对应关系
dict_big_ids = {}
for i, id_raw in enumerate(v_ids):
    dict_big_ids[i] = id_raw    # {new_id: raw_id}
dict_raw_ids = {v: k for k, v in dict_big_ids.items()}    # {raw_id: new_id}



# 读取small_csv
videos_small_path = 'video2tags_small.csv'
videos_small = pd.read_csv(videos_small_path)
v_ids_small = videos_small['video_id'].values


new_videos_small = []
for i in range(len(v_ids_small)):  # 遍历small的每一行
    # 获取small在big中对应的编号（有点拗口（就是big重新编号成了0-10725，所以要从新对应一下获取small的））
    new_videos_small.append(dict_raw_ids[v_ids_small[i]])    # [raw_id1, raw_id2, ...]
print(new_videos_small[:10])
print(len(new_videos_small))
# 写出small的新ids的txt文件
with open('new_videos_small.txt', 'w') as f:
    for new_id in new_videos_small:
        f.write(str(new_id)+'\n')

# 写出prompts_long_small的txt文件
prompts_long_small = [prompts_long[i] for i in new_videos_small]
with open('prompts_long_small.txt', 'w') as f:
    for prompt in prompts_long_small:
        f.write(prompt+'\n')
# 写出prompts_short_small的txt文件
prompts_short_small = [prompts_short[i] for i in new_videos_small]
with open('prompts_short_small.txt', 'w') as f:
    for prompt in prompts_short_small:
        f.write(prompt+'\n')
