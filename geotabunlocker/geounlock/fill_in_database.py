import pandas as pd
import datetime

import re
import json
from zhipuai import ZhipuAI

import pandas as pd
from typing import List, Dict
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def safe_lower(text):
    return re.sub(r'[A-Za-z]', lambda m: m.group(0).lower(), text)


def jaccard_similarity(s1: str, s2: str, n: int = 2) -> float:
    """计算两个字符串基于n-gram的Jaccard相似度"""
    set1 = set([s1[i:i+n] for i in range(len(s1) - n + 1)])
    set2 = set([s2[i:i+n] for i in range(len(s2) - n + 1)])
    if not set1 and not set2:
        return 1.0 if s1 == s2 else 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


def match_cols_heuristic(
        source_cols: List[str], target_cols: List[str], config: dict = None) -> Dict[str, str]:

    if not source_cols or not target_cols:
        return {}
    
    weights = {
        'lev': config['weights'].get('lev', 0.2), 
        'jac': config['weights'].get('jac', 0.2), 
        'cos': config['weights'].get('cos', 0.6), 
    }
    similarity_threshold = config.get('similarity_threshold', 0.5)

    source_cols_clean = [safe_lower(s.strip()) for s in source_cols]
    target_cols_clean = [safe_lower(s.strip()) for s in target_cols]

    all_cols_clean = list(set(source_cols_clean + target_cols_clean))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_cols_clean)
    
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    
    all_cols = list(set(source_cols + target_cols))
    cosine_sim_lookup = {
        (all_cols[i], all_cols[j]): cosine_sim_matrix[i, j]
        for i in range(len(all_cols)) for j in range(len(all_cols))
    }

    matches = {}
    matched_target_cols = set()

    for s_col in source_cols:
        best_match_score = -1.0
        best_match_target = None

        for t_col in target_cols:

            if t_col in matched_target_cols:
                continue

            lev_score = fuzz.ratio(s_col.lower().strip(), t_col.lower().strip()) / 100.0
            
            jac_score = jaccard_similarity(s_col.lower().strip(), t_col.lower().strip())
            
            cos_score = cosine_sim_lookup.get((s_col, t_col), 0.0)

            combined_score = (
                lev_score * weights['lev'] +
                jac_score * weights['jac'] +
                cos_score * weights['cos']
            )

            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_target = t_col

        if best_match_target and best_match_score >= similarity_threshold:
            matches[s_col] = best_match_target
            matched_target_cols.add(best_match_target)
            
    return matches


def match_cols_llm(cols: list, csv_cols: list, api_key: str, config: dict = None):

    max_retry = config.get('max_retry', 3)
    llm_model = config.get('llm_model', 'glm-4-plus')
    
    prompt = f'''
        # 角色
        你是一名顶级的地球化学数据管理专家，擅长解析和匹配来自不同实验室的地质数据表格。

        # 任务
        你的任务是分析两个表格的列名列表，并根据严格的语义和科学标准，创建一个从源列到目标列的最佳匹配映射。

        # 匹配标准与约束
        1.  **语义优先**: 匹配的核心是科学含义，而非简单的文本相似度。
        2.  **通用字段匹配**: 对于如 "ID", "Sample Name", "Location" 等通用字段，寻找其最接近的同义词或缩写。例如，“样品编号”可以匹配“编号”或“Sample_ID”。
        3.  **科学字段匹配（硬性规定）**:
            - **同位素**: 只有当元素符号完全相同时才能匹配。例如，`δ34S` 只能匹配 `δ34S` 或 `δ34S (‰)`，绝对不能匹配 `δ82Se`。
            - **化学元素/化合物**: `Fe` (铁) 只能匹配 `Fe` 或 `Iron`，不能匹配 `Mg` (镁)。
            - **单位**: 单位必须兼容。例如，`ppm` 可以匹配 `mg/kg`，但通常不应匹配 `wt%`。
        4.  **无匹配**: 如果源列在目标表中找不到任何符合上述标准的匹配项，则不应在最终结果中包含它。

        # 输出格式
        1.  **思考过程**: 首先，在 `<thinking>` 标签内，简要分析每一对潜在的匹配，并解释为什么匹配或不匹配。
        2.  **最终结果**: 其次，提供一个严格的JSON字典，格式为 `{{"源列名": "目标列名"}}`。

        # 示例
        <thinking>
        - "样品编号" vs "编号": 语义完全一致，都是指样品的唯一标识符。匹配。
        - "采样位置" vs "采样地点": 语义完全一致，都指样品采集的地理位置。匹配。
        - "δ34s/‰" vs "δ82/76Se (‰)": 虽然格式都是同位素千分差，但科学元素不同（S vs Se）。根据规则3，这是硬性不匹配。
        </thinking>

        {{"样品编号": "编号", "采样位置": "采样地点"}}

        # 待处理任务
        现在，请根据以上所有规则，处理以下任务：

        **表格1 (源):** {cols}
        **表格2 (目标):** {csv_cols}

        必须按照`输出格式`输出结果！
        回答的`源列名`与`目标列名`必须与给定的`源列名`与`目标列名`保持一致，不得私自修改！
        如果不存在如何`源列名`与`目标列名`匹配，则返回`{{}}`
    '''

    client = ZhipuAI(api_key=api_key)

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response.choices[0].message.content.strip()
            response_content = response_content.replace('\\n', '').replace('\\N', '').replace('\\t', '')
            try:
                result_dict = json.loads(response_content)
                if isinstance(result_dict, dict): return result_dict
                print(f"错误：LLM返回的不是一个JSON字典。\n{response_content}")
            except json.JSONDecodeError:
                match = re.search(r'```json\s*({.*?})\s*```', response_content, re.DOTALL)
                if match:
                    json_string = match.group(1)
                    try:
                        result_dict = json.loads(json_string)
                        if isinstance(result_dict, dict): return result_dict
                        print(f"错误：正则匹配到的不是一个JSON字典。\n{result_dict}")
                    except json.JSONDecodeError: 
                        print(f"错误：正则匹配到的JSON字符串解析失败。\n{json_string}")
                else: 
                    fallback_match = re.search(r'{.*}', response_content, re.DOTALL)
                    if fallback_match:
                        try:
                            result_dict = json.loads(fallback_match.group(0))
                            if isinstance(result_dict, dict): return result_dict
                        except json.JSONDecodeError:
                            print(f"错误：最终回退匹配也无法解析JSON。\n{response_content}")
                    else:
                        print(f"错误：在返回内容中未找到任何有效的JSON格式。\n{response_content}")

        except Exception as e:
            print(f"调用LLM时发生错误 (重试 {attempt + 1}/{max_retry}): {e}")

    print("错误：LLM调用和解析多次失败，返回空字典。")
    return {}


def get_df_match(df: pd.DataFrame, df_csv: pd.DataFrame, match_cols: dict, metedata: dict, line_num: int = 0):

    df_temp = pd.DataFrame(
        {col: pd.NA for col in df_csv.columns},
        index=range(len(df))
    )

    ship_col = ['采样地点', '样品号', '测试矿物/岩石', '测试方法', '测试岩石/矿物名称', '测试矿物']
    if set(match_cols.values()).issubset(set(ship_col)):
        return df_csv

    for col, sheet_col in match_cols.items():
        if col in df.columns:
            df_temp[sheet_col] = df[col]

    source = metedata['source'] if 'source' in metedata.keys() else '无'
    sample_location = metedata['sample_location'] if 'sample_location' in metedata.keys() else ''

    df_temp["编号"] = [i for i in range(line_num + 1, line_num  + 1 + len(df_temp))]
    df_temp["保存时间"] = [datetime.datetime.now()] * len(df_temp)
    df_temp["资料来源/参考文献"] = [source] * len(df_temp)
    if '采样地点' in df_temp.columns:
        df_temp['采样地点'] = sample_location + df_temp['采样地点'].astype(str)

    return df_temp


import os

def data_fill(
    df: pd.DataFrame,
    all_csv_data: dict,
    metedata: dict, 
    config: dict,
    api_key: str = None,
):
    cols = df.columns.to_list()
    if not cols: return
    print(f'cols: {cols}')

    map_database = config.get('map_database', '')
    with open(map_database, 'r', encoding='utf-8') as map_file:
        colmap_database = json.load(map_file)
    
    table_all_map = {}

    for i, (csv_path, df_csv) in enumerate(all_csv_data.items()):
        with open(csv_path, 'r', encoding='utf-8') as f:
            line_num = sum(1 for line in f) - 1
        csv_name = os.path.basename(csv_path)
        colmap = colmap_database.setdefault(csv_name, {})

        csv_cols = df_csv.columns.tolist()
        csv_cols = [x for x in csv_cols if x not in ['编号', '资料来源/参考文献', '修改时间']]
        
        cols_need_match = []
        csv_cols_need_match = []
        match_cols = {}

        for col in cols:
            if col in colmap.keys():
                if colmap[col] == 'no_match': continue
                for csv_col in csv_cols:
                    if csv_col in colmap[col]:
                        match_cols[col] = csv_col
            else:
                cols_need_match.append(col)

        csv_cols_need_match = [x for x in csv_cols if x not in match_cols.values()]

        new_match_cols = {}
        if cols_need_match:
            if api_key:
                new_match_cols = match_cols_llm(cols_need_match, csv_cols_need_match, api_key, config.get('llm', {}))
            else:
                new_match_cols = match_cols_heuristic(cols_need_match, csv_cols_need_match, config.get('heuristic', {}))

        if match_cols or new_match_cols:
            match_cols.update(new_match_cols)
            for col in cols:
                if col in colmap.keys() and col in match_cols.keys():
                    csv_col = match_cols[col]
                    if csv_col not in colmap[col]:
                        colmap[col].append(csv_col)
                elif col not in colmap.keys() and col in match_cols.keys():
                    csv_col = match_cols[col]
                    colmap[col] = [csv_col]
                elif col not in colmap.keys() and col not in match_cols.keys():
                    colmap[col] = 'no_match'

            colmap_database[csv_name] = colmap

            df_match = get_df_match(df, df_csv, match_cols, metedata, line_num)
            if not df_match.empty:
                df_match.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                print(f"{i} CSV: {csv_name}")
                print(f"    匹配字段: {match_cols}")
                table_all_map.update(match_cols)

    with open(map_database, 'w', encoding='utf-8') as map_file:
        json.dump(colmap_database, map_file, ensure_ascii=False, indent=2)
        
    print("表格信息已整合完成！\n")

    return table_all_map



def data_fill_human_modify(
    df: pd.DataFrame,
    all_csv_data: dict,
    metedata: dict, 
    config: dict,
):  
    print('-- 填入人工匹配数据 --')
    cols = df.columns.to_list()
    if not cols: return
    print(f'cols: {cols}')

    human_colmap = metedata['col_map']

    map_database = config.get('map_database', '')
    with open(map_database, 'r', encoding='utf-8') as map_file:
        colmap_database = json.load(map_file)
    
    table_all_map = {}

    for i, (csv_path, df_csv) in enumerate(all_csv_data.items()):
        with open(csv_path, 'r', encoding='utf-8') as f:
            line_num = sum(1 for line in f) - 1

        csv_name = os.path.basename(csv_path)

        colmap = colmap_database.setdefault(csv_name, {})

        csv_cols = df_csv.columns.tolist()
        csv_cols = [x for x in csv_cols if x not in ['编号', '资料来源/参考文献', '修改时间']]
        
        match_cols = {}
        for col in cols:
            if col in human_colmap.keys():
                if human_colmap[col] == 'no_match': continue
                for csv_col in csv_cols:
                    if csv_col == human_colmap[col]:
                        match_cols[col] = csv_col
        if match_cols:
            for col in cols:
                if col in colmap.keys() and col in match_cols.keys():
                    if colmap[col] == 'no_match' and match_cols[col] != 'no_match': 
                        colmap[col] = [match_cols[col]]
                    elif colmap[col] == 'no_match' and match_cols[col] == 'no_match': 
                        colmap[col] = 'no_match'
                    elif colmap[col] != 'no_match' and match_cols[col] == 'no_match':
                        colmap[col] = 'no_match'
                    elif colmap[col] != 'no_match' and match_cols[col] != 'no_match':    
                        if match_cols[col] not in colmap[col]:
                            colmap[col].append(csv_col)
                            colmap[col] = [match_cols[col]]
                elif col not in colmap.keys() and col in match_cols.keys():
                    colmap[col] = [match_cols[col]]
                elif col not in colmap.keys() and col not in match_cols.keys():
                    colmap[col] = 'no_match'
                elif col in colmap.keys() and col not in match_cols.keys():
                    colmap[col] = 'no_match'

            colmap_database[csv_name] = colmap

            df_match = get_df_match(df, df_csv, match_cols, metedata, line_num)
            if not df_match.empty:
                df_match.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                print(f"{i} CSV: {csv_name}")
                print(f"    匹配字段: {match_cols}")
                table_all_map.update(match_cols)
    
    with open(map_database, 'w', encoding='utf-8') as map_file:
        json.dump(colmap_database, map_file, ensure_ascii=False, indent=2)
        
    print("表格信息已重新整合完成！")

    return table_all_map

