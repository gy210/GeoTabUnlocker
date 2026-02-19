import os
import json
from typing import List, Optional, Dict

from zhipuai import ZhipuAI

def determine_title_with_llm(
    candidate_texts: List[str], 
    api_key: str,
    config: Dict = None
) -> Optional[str]:
    if not candidate_texts:
        return None

    if config is None:
        config = {}
    model_name = config.get('model', 'glm-4-plus')
    max_retries = config.get('max_retries', 3)

    formatted_candidates = json.dumps(candidate_texts, ensure_ascii=False)

    prompt = f"""
        # 角色
        你是一名经验丰富的科研编辑和文献计量学专家，擅长从混杂的文本中精确识别出学术论文的标题。

        # 任务
        分析以下从文献数据中提取的文本片段列表，并从中选择**唯一一个**最能代表**论文标题**的字符串。

        # 严格规则
        1.  **识别标准**：论文标题通常是一个描述研究**主题、方法或发现**的完整短语或句子。它通常是列表中最长、信息最丰富的字符串之一。
        2.  **必须忽略以下内容**（这些绝对不是论文标题）：
            -   **期刊或书籍名称** (例如: "Nature", "Geochimica et Cosmochimica Acta", "Economic Geology")
            -   **作者姓名** (例如: "Wang et al.", "Li, S. J.", "张三")
            -   **出版年份** (例如: "2023", "1998")
            -   **卷、期、页码** (例如: "Vol. 123", "45(3)", "pp. 11-25")
            -   **机构或大学名称** (例如: "中国科学院", "Stanford University")
            -   **文档类型** (例如: "博士学位论文", "研究报告", "未发表数据")
        3.  **返回原始字符串**：你选择的答案必须是候选列表中一模一样的原始字符串，不得进行任何修改。
        4.  **无合适标题**：如果列表中没有任何一个字符串符合论文标题的标准，则判定为找不到标题。
        5.  **如果既有中文标题，又有相应英文标题，则以中文标题为主**

        # 待分析的候选列表
        {formatted_candidates}

        # 输出格式
        你的输出必须严格遵守以下JSON格式，不要包含任何额外的解释或文本。

        - 如果找到了合适的标题:
        {{"title": "选择的论文标题字符串"}}

        - 如果没有找到任何合适的标题:
        {{"title": null}}

        # 示例
        - 输入: ["Geology of the Candelaria-Punta del Cobre district, Chile", "Economic Geology", "1998", "Ryan, P.J. et al."]
        - 输出: {{"title": "Geology of the Candelaria-Punta del Cobre district, Chile"}}

        - 输入: ["Geochimica et Cosmochimica Acta", "2015", "Li, S. et al."]
        - 输出: {{"title": null}}

        请根据以上规则，给出你的最终判断。
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
            if isinstance(parsed_json, dict) and 'title' in parsed_json:
                title = parsed_json['title']
                if title is None or title in candidate_texts:
                    return title
                else:
                    print(f"警告：LLM返回了一个不在原始列表中的标题: '{title}'")
                    return title

        except Exception as e:
            print(f"调用LLM时发生错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("错误：LLM调用失败，已达到最大重试次数。")
                return None
    
    return None