from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

from zhipuai import ZhipuAI


def is_complex_table(table):
    cells = table.find_all(['td', 'th'])
    for cell in cells:
        if 'rowspan' in cell.attrs or 'colspan' in cell.attrs:
            return True
    return False


def parse_complex_table(table):
    grid = []
    
    for row_id, tr in enumerate(table.find_all('tr')):
        grid.append([])
        for td in tr.find_all(['td', 'th']):
            while len(grid) <= row_id:
                grid.append([])

            col_id = 0
            while len(grid[row_id]) > col_id and grid[row_id][col_id] is not None:
                col_id += 1
            
            colspan = int(td.get('colspan', 1))
            rowspan = int(td.get('rowspan', 1))
            
            cell_content = split_caps_and_lower(td.get_text(strip=True))
            
            for r in range(rowspan):
                cur_row = row_id + r
                while len(grid) <= cur_row:
                    grid.append([None] * (col_id + colspan))
                
                while len(grid[cur_row]) < col_id + colspan:
                    grid[cur_row].append(None)
                    
                for c in range(colspan):
                    grid[cur_row][col_id + c] = cell_content
    
    max_cols = max(len(row) for row in grid) if grid else 0
    for row in grid:
        while len(row) < max_cols:
            row.append('')

    html_convert = ''
    for row in grid:
        html_convert += '<tr>'
        for col in row:
            html_convert += '<td>' + col + '</td>'
        html_convert += '</tr>'

    html_convert = f"<table>{html_convert}</table>"
            
    return pd.DataFrame(grid), html_convert


def split_caps_and_lower(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    text = text.replace(',', '_').replace('，', '_')
    return text.lower()


def clean_table_data(df):

    df = df.replace('', np.nan)

    df = df.infer_objects()

    df = df.dropna(how='all').dropna(axis=1, how='all')

    return df


def html_table_to_excel(html: str, metadata: dict, output_dir: Path):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    for table in tables:
        df, html_convert = parse_complex_table(table)
        df = clean_table_data(df)
    return df, html_convert