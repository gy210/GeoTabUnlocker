import os
import json
from typing import List, Dict, Union

from zhipuai import ZhipuAI

def extract_location_from_title(
    table_title: str, 
    api_key: str,
    config: Dict = None
) -> List[str]:
    if not table_title or not table_title.strip():
        return ''

    if config is None:
        config = {}
    model_name = config.get('model', 'glm-4')
    max_retries = config.get('max_retries', 2)

    prompt = f"""
        # 角色
        你是一位顶尖的地质学家和文献分析专家，你的任务是从技术表格的标题中，精确地抽取出所有与**地理采样位置**相关的信息。

        # 任务
        分析以下地质表格的标题，并提取出其中明确提到的所有地理位置、矿床、山脉、地块或特定地质构造区。

        # 严格规则
        1.  **提取对象**：只提取地理名称。例如："大别山"、"东太平洋海隆"、"金川镍矿"、"华北克拉通"。
        2.  **必须忽略**：
            -   矿物或岩石名称 (如: "玄武岩", "辉长岩", "锆石")
            -   化学元素或同位素 (如: "主量元素", "稀土元素", "Re-Os同位素")
            -   分析方法或数据类型 (如: "电子探针分析结果", "地球化学数据")
            -   抽象的地质概念 (如: "俯冲带", "地幔柱"，除非它指向一个具体的名字，例如"夏威夷地幔柱")
        3.  **输出格式**: 你的输出必须是严格的JSON格式，包含一个名为 "locations" 的键，其值为一个**字符串列表**。
        4.  **多个地点**: 如果标题中包含多个地点，请将它们全部提取到列表中。
        5.  **无地点**: 如果标题中不包含任何地理位置信息，请返回一个**空列表**。

        # 待分析的标题
        "{table_title}"

        # 输出格式示例
        - 输入: "表1. 大别山榴辉岩样品主量元素分析结果 (wt.%)"
        - 输出: {{"locations": ["大别山"]}}

        - 输入: "Table 2. Geochemical data for samples from the Dabie and Sulu orogens"
        - 输出: {{"locations": ["Dabie orogen", "Sulu orogen"]}}

        - 输入: "附表3. 锆石U-Pb年龄和Hf同位素组成"
        - 输出: {{"locations": []}}

        请根据以上规则，给出你的最终提取结果。不要包含任何额外的解释或文本。
    """

    client = ZhipuAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            
            parsed_json = json.loads(response_content)
            if (isinstance(parsed_json, dict) and 
                'locations' in parsed_json and 
                isinstance(parsed_json['locations'], list)):
                return ''.join(parsed_json['locations'])
            else:
                 print(f"警告：LLM返回了格式不正确的JSON: {response_content}")

        except Exception as e:
            print(f"调用LLM时发生错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("错误：LLM调用失败，已达到最大重试次数。")
    
    return ''