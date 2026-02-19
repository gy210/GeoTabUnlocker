import io
import pandas as pd
from typing import List, Tuple

from pathlib import Path
from zhipuai import ZhipuAI


def detect_header_rows(df: pd.DataFrame, max_header_rows: int = 3) -> Tuple[List[int], int]:
    n_rows = len(df)
    max_check = min(max_header_rows, n_rows)
    
    if max_check < 1:
        return [], 0

    df_clean = df.iloc[:max_check].astype(str).fillna("").replace("nan", "").values
    n_cols = df_clean.shape[1]

    if n_cols == 0:
        return [], 0

    parent_scores = []
    numeric_ratios = []

    for i in range(max_check):
        row = df_clean[i]
        num_ratio = sum(1 for x in row if x.replace('.', '').replace('±', '').replace('～', '').replace('-', '').isdigit()) / n_cols
        numeric_ratios.append(num_ratio)

        expand_score = 0
        if i < max_check - 1:
            next_row = df_clean[i + 1]
            j = 0
            while j < n_cols:
                cell = row[j]
                if not cell:
                    j += 1
                    continue
                k = j
                while k < n_cols and row[k] == cell:
                    k += 1
                block_length = k - j
                if block_length >= 2:
                    next_vals_in_block = next_row[j:k]
                    if len(set(next_vals_in_block)) > 1:
                        expand_score += block_length 
                j = k
        parent_scores.append(expand_score)

    header_rows = []
    for i in range(max_check):
        is_header = False

        if parent_scores[i] > 0 and numeric_ratios[i] < 0.2 and any(df_clean[i]):
            is_header = True

        if not is_header and i < max_check and i > 0:
            current_unique = len(set(df_clean[i]))
            next_unique = len(set(df_clean[i-1]))
            overlap = len(set(df_clean[i]) & set(df_clean[i-1]))
            if overlap >= 1:
                is_header = True

        if is_header:
            header_rows.append(i)
        else:
            break
    if not header_rows:
        header_rows = [0]

    data_start_row = max(header_rows) + 1
    if data_start_row >= n_rows:
        data_start_row = max(header_rows)

    return header_rows, data_start_row



def build_flat_columns(df: pd.DataFrame, header_rows: List[int]) -> List[str]:
    if not header_rows:
        return [f"col_{i}" for i in range(df.shape[1])]

    headers = df.iloc[header_rows].astype(str).fillna("").replace("nan", "").values
    n_levels, n_cols = headers.shape

    flat_names = []
    for j in range(n_cols):
        path = []
        for i in range(n_levels):
            val = headers[i, j].strip()
            if not val:
                continue
            if not path or val != path[-1]:
                path.append(val)
        flat_names.append("-".join(path) if len(path) > 1 else val)
    
    return flat_names



def header_parser_heuristic(df_raw: pd.DataFrame):
    header_rows, data_start = detect_header_rows(df_raw)
    print(f"✅ 检测到表头行: {header_rows}, 数据起始行: {data_start}")

    flat_cols = build_flat_columns(df_raw, header_rows)
    print(f"✅ 扁平列名: {flat_cols}")

    df_clean = df_raw.iloc[data_start:].copy()
    df_clean.columns = flat_cols
    print("\n✅ 扁平化列字段表格:")
    print(df_clean.to_string(index=False))

    return flat_cols, df_clean




import re
import pandas as pd
from io import StringIO

def sanitize_llm_response(text: str) -> str:
    text = text.strip()
    
    csv_match = re.search(r"```(?:csv|text)?\s*\n(.*?)\n```", text, re.DOTALL)
    if csv_match:
        text = csv_match.group(1)
    
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.count(',') >= 2: 
            text = '\n'.join(lines[i:])
            break
    
    return text


def parse_csv_with_fallback(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(StringIO(text))
        if len(df.columns) > 1:
            return df
    except Exception as e:
        print(e)
        pass

    for sep in ['\t', ';']:
        try:
            df = pd.read_csv(StringIO(text), sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception as e:
            print(e)
            pass

    lines = text.strip().splitlines()
    if not lines:
        return pd.DataFrame()

    rows = []
    for line in lines:
        row = [x.strip().strip('"').strip("'") for x in line.split(',')]
        rows.append(row)
    
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    
    if len(rows) > 0:
        header = rows[0]
        data = rows[1:]
        return pd.DataFrame(data, columns=header)
    
    return pd.DataFrame()


def repair_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.columns.tolist() == ['Unnamed: 0']:
        return df

    if df.columns.tolist() == ['Unnamed: 0'] or all(str(c).startswith('Unnamed') for c in df.columns):
        first_row = df.iloc[0].astype(str)
        numeric_ratio = sum(x.replace('.', '').isdigit() for x in first_row) / len(first_row)
        if numeric_ratio < 0.3: 
            new_cols = [str(x).strip() for x in first_row]
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = new_cols
            return df

    df.columns = [
        str(col).strip().replace('\ufeff', '').replace('"', '').replace("'", "") 
        for col in df.columns
    ]
    return df



def header_parser_llm(html: str, api_key: str, metadata: dict = None):

    client = ZhipuAI(api_key=api_key)

    separator = '-'
    output_format = 'CSV'

    prompt = f'''
        # 角色扮演 (Persona)
        你是一位顶级的自动化数据工程师。你的核心任务是将提供的高度复杂的、为人类阅读而设计的HTML表格序列，精确地转换为机器可读的、干净的结构化数据。

        # 核心任务 (Mission)
        你的目标是将一个包含嵌套表头（`rowspan` 和 `colspan`）的HTML表格，转换成一个扁平化的、单一表头的表格，并以指定的格式输出。你必须严格遵循下面的处理逻辑和输出格式要求。

        # 核心处理逻辑 (Core Processing Logic)
        1. **动态表头检测**
            你必须自动确定表头与数据之间的边界。请使用以下特征来区分：
            *   **表头行 (Header Rows)** 的特征：通常包含描述性的标签、类别或单位（如“年份”、“销售额”、“%”）；经常使用 `<td>` 标签；并且频繁利用 `colspan` 和 `rowspan` 属性来创建复杂的层级结构。
            *   **数据行 (Data Rows)** 的特征：通常包含具体的数值、专有名词、日期等实际数据点；结构相对规整、重复性强，很少包含 `colspan` 或 `rowspan`。

            **关键指令：你的首要任务是找到第一个明显属于“数据行”的行。所有在该行之前的所有行，都应被视为层级表头部分。**
        2.  **解析层级关系**:
            *   对于跨列的单元格 (`colspan`)，其内容是它下方所有对应列的**上级标签**。
            *   对于跨行的单元格 (`rowspan`)，其内容是它右侧所有对应行的**通用标签**。
        3.  **扁平化表头**:
            *   为最终的每一列生成一个**单一的、描述性的标签**。
            *   生成规则是：将该列从上到下的所有层级标签用指定的 `{separator}` 连接起来。
            *   如果某列只有一个层级（例如，一个跨越所有表头行的单元格），则直接使用该单元格的文本作为标签。
        4.  **数据清洗**:
            *   提取所有单元格的纯文本内容。
            *   去除每个单元格内容首尾的空白字符。

        # 输出格式要求 (Output Format)
        *   将最终结果以 **{output_format}** 格式输出。
        *   **不要包含任何额外的解释、评论或代码块标记。** 你的输出应该直接是纯粹的数据内容。

        ---
        ### 样例 (Few-shot Example)
        这是你需要学习的一个完美范例。严格按照这个范例的逻辑和格式进行处理。

        **[样例输入]**

        ```html
            <table>
                <tr><td rowspan="2">岩石</td><td colspan="4">铅同位素组成/%</td><td colspan="5">铅同位素比值/%</td></tr>
                <tr><td>204Pb</td><td>206Pb</td><td>207Pb</td><td>208Pb</td><td>206Pb/204Pb</td><td>207Pb/204Pb</td><td>208Pb/204Pb</td><td>206Pb/207Pb</td><td>206Pb/208Pb</td></tr>
                <tr><td>非矿伟晶岩(3)*</td><td>1.25</td><td>32.01</td><td>19.17</td><td>47.66</td><td>25.608</td><td>15.336</td><td>38.128</td><td>1.700</td><td>0.672</td></tr>
                <tr><td>含矿伟晶岩(11)*</td><td>0.425</td><td>73.238</td><td>9.754</td><td>16.913</td><td>172.325</td><td>22.951</td><td>39.795</td><td>7.509</td><td>4.330</td></tr>
            </table>

        **[样例输出]**

        岩石,铅同位素组成/%-204Pb,铅同位素组成/%-206Pb,铅同位素组成/%-207Pb,铅同位素组成/%-208Pb,铅同位素比值/%-206Pb/204Pb,铅同位素比值/%-207Pb/204Pb,铅同位素比值/%-208Pb/204Pb,铅同位素比值/%-206Pb/207Pb,铅同位素比值/%-206Pb/208Pb
        非矿伟晶岩(3)*,1.25,32.01,19.17,47.66,25.608,15.336,38.128,1.700,0.672
        含矿伟晶岩(11)*,0.425,73.238,9.754,16.913,172.325,22.951,39.795,7.509,4.330

        # 下面是HTML表格序列，请你输出结果：
        ```html
            {html}
    '''

    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[{"role": "user", "content": prompt}],
    )
    llm_output = response.choices[0].message.content

    clean_text = sanitize_llm_response(llm_output)
    
    df = parse_csv_with_fallback(clean_text)
    
    df = repair_headers(df)

    print(f"✅ 扁平列名: {df.columns.to_list()}")

    print("\n✅ 扁平化列字段表格:")
    print(df.to_string(index=False))

    return df


def header_parser(df, metadata: dict=None, api_key: str = None):
    if api_key:
        print("大模型解析表头...")
        df = header_parser_llm(metadata['html_convert'], api_key, metadata)
        flat_cols = df.columns.to_list()

    else:
        print("启发式方法解析表头...")
        flat_cols, df = header_parser_heuristic(df)

    return df, flat_cols