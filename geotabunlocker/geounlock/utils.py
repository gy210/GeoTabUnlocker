import math
import torch
import numpy as np

import cv2
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import pandas as pd

from typing import Dict, Any, List

from geounlock.data import ABSOLUTE_TEXT


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    print(f"正在将PDF '{pdf_path}' 转换为图片 (DPI={dpi})...")
    pdf_doc = fitz.open(pdf_path)
    images = []
    
    for page in pdf_doc.pages():
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace='rgb', alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    print(f"转换完成，共 {len(images)} 页。")
    return images



def custom_format_html(html_string, tokenizer):
    """Custom format html string"""
    tokens_to_remove = [
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        "<s_answer>",
        "</s_answer>",
    ]
    for token in tokens_to_remove:
        html_string = html_string.replace(token, "")

    html_seq = "<html><body><table>" + html_string + "</table></body></html>"

    return html_string, html_seq



def decode_OTSL_seq(otsl_token_seq, pointer_tensor, cell_text_data):
    cell_text = None
    OTSL_full_compilation = []
    OTSL_row_compilation = []
    curr_column_index = 0

    for data_ind, token in enumerate(otsl_token_seq):
        if token == "C-tag":
            mapping_mask = pointer_tensor[data_ind]

            coord_indices = torch.nonzero(mapping_mask).squeeze(-1)
            if len(coord_indices) == 0: 
                cell_text = None
            else:
                indices_list = coord_indices.tolist()
                for coord_ind in indices_list:
                    if coord_ind == 0: continue
                    elif coord_ind > len(cell_text_data): continue
                    else:
                        if cell_text is None:
                            cell_text = cell_text_data[coord_ind - 1]
                        else:
                            cell_text += " " + cell_text_data[coord_ind - 1]

            OTSL_row_compilation.append([1, 0, 0, cell_text])
            curr_column_index += 1
            cell_text = None
        elif token == "NL-tag":
            OTSL_full_compilation.append(OTSL_row_compilation)
            OTSL_row_compilation = []
            curr_column_index = 0
        elif token == "L-tag":
            for col_i in range(len(OTSL_row_compilation)):
                col_i_value = OTSL_row_compilation[-1 - col_i]
                if col_i_value is not None:
                    col_i_value[2] += 1
                    break
            OTSL_row_compilation.append(None)
            curr_column_index += 1

        elif token == "U-tag":
            for row_i in range(len(OTSL_full_compilation)):
                row_i_value = OTSL_full_compilation[-1 - row_i]
                if (
                    curr_column_index < len(row_i_value)
                    and row_i_value[curr_column_index] is not None
                ):
                    row_i_value[curr_column_index][1] += 1
                    break

            OTSL_row_compilation.append(None)
            curr_column_index += 1
        elif token == "X-tag":
            OTSL_row_compilation.append(None)
            curr_column_index += 1
            continue
        else: continue

    if len(OTSL_row_compilation) > 0:
        OTSL_full_compilation.append(OTSL_row_compilation)

    OTSL_full_compilation = [
        item for sublist in OTSL_full_compilation for item in sublist
    ]
    output_html_seq = "<tr>"
    current_data_index = 0
    for i, token in enumerate(otsl_token_seq):
        if token in ["L-tag", "U-tag", "X-tag"]:
            current_data_index += 1
            continue
        elif token == "C-tag":
            cell_info = OTSL_full_compilation[current_data_index]
            if cell_info is not None:
                if cell_info[1] == 0 and cell_info[2] == 0:
                    if cell_info[3] is None:
                        output_html_seq += "<td></td>"
                    else:
                        output_html_seq += "<td>" + cell_info[3] + "</td>"

                elif cell_info[1] == 0:
                    if cell_info[3] is None:
                        output_html_seq += '<td colspan="%s"></td>' % (cell_info[2] + 1)
                    else:
                        output_html_seq += '<td colspan="%s">' % (cell_info[2] + 1) + cell_info[3] + "</td>"

                elif cell_info[2] == 0:
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s"></td>' % (cell_info[1] + 1)
                    else:
                        output_html_seq += '<td rowspan="%s">' % (cell_info[1] + 1) + cell_info[3] + "</td>"

                else: 
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s" colspan="%s"></td>' % (cell_info[1] + 1, cell_info[2] + 1)
                    else:
                        output_html_seq += (
                            '<td rowspan="%s" colspan="%s">' % (cell_info[1] + 1, cell_info[2] + 1)
                            + cell_info[3]
                            + "</td>"
                        )

            current_data_index += 1

        elif token == "NL-tag":
            output_html_seq += "</tr><tr>"
        
        elif token in ["[DEC]", "[SEP]"]: continue

        else:
            if token == "▁":
                token_to_add = " "
                output_html_seq += token_to_add
            else:
                token_to_add = token.replace("▁", "")
                output_html_seq += token_to_add

    tmp_split = output_html_seq.rsplit("<tr>", 1)
    output_html_seq = tmp_split[0] + tmp_split[1]

    output_html_seq = output_html_seq.replace("<pad>", "")

    output_html_seq = f"<table>{output_html_seq}</table>"

    return output_html_seq



def clean_and_convert_data_robust(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(how='all').dropna(axis=1, how='all')
    df = df.drop_duplicates().reset_index(drop=True)

    df = merge_complement_rows(df)
    df = split_row(df)

    new_columns_map = {}

    for col in df.columns:

        for keyword in ABSOLUTE_TEXT:
            if keyword.lower() in col.lower():
                continue
        
        series = df[col].astype(str)
        
        is_percent_col = False
        if '%' in col:
            is_percent_col = True
        else:
            non_na_series = series.dropna()
            if not non_na_series.empty:
                try:             
                    percent_count = non_na_series.str.contains('%', na=False).sum()
                except Exception as e:
                    print(f"错误：表格序列预测不准确, 存在重复列字段或空列字段！")
                    print(df.to_string(index=False))
                    return pd.DataFrame()
                    

                if (percent_count / len(non_na_series)) > 0.3:
                    is_percent_col = True

        if is_percent_col:
            try:
                numeric_part = series.str.extract(r'(-?\d+\.?\d*)', expand=False)
                numeric_val = pd.to_numeric(numeric_part, errors='coerce')

                df[col] = numeric_val * 10.0

                new_col_name = col.replace('%', '‰').replace('（', '(').replace('）', ')')
                new_col_name = new_col_name.replace(' ', '')
                if '‰' not in new_col_name:
                    new_col_name = f"{col}(‰)"
                new_columns_map[col] = new_col_name
            except Exception as e:
                print(f"警告：处理百分比列 '{col}' 时出现问题: {e}")
        else:
            try:
                converted_series = pd.to_numeric(series, errors='coerce')
                conversion_ratio = converted_series.notna().sum() / len(series.dropna())

                if conversion_ratio == 1.0:
                    df[col] = converted_series
                new_col_name = col.replace('{', '(').replace('}', ')').replace('（', '(').replace('）', ')')   
                new_col_name = new_col_name.replace(' ', '')
                new_columns_map[col] = new_col_name
                
            except Exception as e:
                print(f"警告：尝试转换数值列 '{col}' 时出现问题，已保持原样: {e}")

    df = df.rename(columns=new_columns_map)
    print("\n✅ 清洗后表格:")
    print(df.to_string(index=False))

    return df


def merge_complement_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    
    rows_to_drop = []
    
    for i in range(len(df_processed) - 1):
        if i in rows_to_drop:
            continue
            
        current_row = df_processed.iloc[i]
        next_row = df_processed.iloc[i + 1]
        
        is_na_in_current = current_row.isna()
        is_not_na_in_current = ~is_na_in_current

        condition1 = next_row[is_na_in_current].notna().all()
        condition2 = next_row[is_not_na_in_current].isna().all()

        if condition1 and condition2:
            print(f"发现互补行：索引 {i} 和 {i+1}。正在合并...")
            df_processed.loc[i, is_na_in_current] = next_row[is_na_in_current]
            rows_to_drop.append(i + 1)

    if rows_to_drop:
        print(f"删除已被合并的行，索引为: {rows_to_drop}")
        df_processed = df_processed.drop(index=rows_to_drop)

    return df_processed.reset_index(drop=True)


def is_multiple_of(a1, a2):

    arr1 = np.array(a1)
    arr2 = np.array(a2)
    
    if np.any(arr2 == 0): return (False, None, False, None)
    
    ratios = arr1 / arr2
    
    if np.allclose(ratios, ratios[0]) and ratios[0] > 1:
        n = round(ratios[0])
        if abs(ratios[0] - n) < 1e-5:
            return (True, n, False, None)

    if sum(arr1 > arr2) / len(arr1) >= 0.7:
        values, counts = np.unique(ratios, return_counts=True)
        mode_index = np.argmax(counts)
        mode = int(values[mode_index])
        return (False, None, True, mode)
    
    return (False, None, False, None)

def split_row(df: pd.DataFrame, separator: str = ' ') -> pd.DataFrame:
    new_rows = []
    for i in range(len(df)):
        curr_counts = [len(str(item).split(separator)) for item in df.iloc[i]]
        if i == 0:
            last_counts = curr_counts
            next_counts = [len(str(item).split(separator)) for item in df.iloc[i+1]]
        elif i == len(df) - 1:
            last_counts = [len(str(item).split(separator)) for item in df.iloc[i-1]]
            next_counts = curr_counts
        else:
            last_counts = [len(str(item).split(separator)) for item in df.iloc[i-1]]
            next_counts = [len(str(item).split(separator)) for item in df.iloc[i+1]]


        merge_tag_1, n_1, may_merge_tag_1, m_1 = is_multiple_of(curr_counts, last_counts)
        merge_tag_2, n_2, may_merge_tag_2, m_2 = is_multiple_of(curr_counts, next_counts)
        if merge_tag_1 or merge_tag_2:
            print(f"发现并拆分已确认的合并行，索引为 {i}...")
            n = None
            if n_1 == n_2: n = n_1
            elif n_1: n = n_1
            elif n_2: n = n_2

            rows = []
            for item in df.iloc[i]:
                row_data = []
                for data in str(item).split(separator)[::n]:
                    row_data.append(data)
                rows.append(row_data)
            rows = list(zip(*rows))
            new_rows.extend(rows)
        
        elif may_merge_tag_1 or may_merge_tag_2:
            m = None
            if m_1 == m_2: m = m_1
            elif m_1: m = m_1
            elif m_2: m = m_2

            rows = []
            for item in df.iloc[i]:
                row_data = str(item).split(separator)
                if len(row_data) > m: row_data = row_data[:m]
                elif len(row_data) < m: row_data.extend([row_data[-1]] * (m - len(row_data)))
                rows.append(row_data)
            rows = list(zip(*rows))
            new_rows.extend(rows)
        
        else:
            new_rows.append(df.iloc[i].tolist())
            
    return pd.DataFrame(new_rows, columns=df.columns)