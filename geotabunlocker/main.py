from typing import List, Dict, Sequence

import torch
import pandas as pd

import os
import json
import jsonlines
import gradio as gr
import logging

logging.basicConfig(
    format='%(asctime)s  | INFO    |    %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO
)

from geounlock.pdf_processor_main import get_pipeline, DocumentExtractionPipeline
from geounlock.fill_in_database import data_fill_human_modify
from tool import excel_to_csv, csvs_to_excel
from css import CSS



init_colmap = {'Null': 'Null'}
def dict2dataframe(d):
    return pd.DataFrame(list(d.items()), columns=["表格字段", "数据库字段"])

pipeline = get_pipeline()


with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# 📄 地球化学表格数据提供工具 🌟",elem_classes=["title"])
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 自动化部分",elem_classes=["title"])
            with gr.Row():
                pdf_path = gr.File(label="选择输入文件(建议PDF数量小于10)")
                download_file = gr.File(label="下载文件")
            status_message = gr.Markdown()
            with gr.Row():
                clean_button = gr.Button("清空界面", elem_classes=["clean-button"])
                gr.Textbox(visible=False,scale=0, min_width=150)
                submit_button = gr.Button("提交", variant="primary")

        with gr.Column(scale=0, min_width=50):
            gr.HTML("""
                <div style="
                    border-left: 2px solid #ccc;
                    height: 100%;
                    margin: 0 10px;
                "></div>""")

        with gr.Column(scale=1, min_width=500):
            gr.Markdown("### 人工审验部分",elem_classes=["title"])
            table_image = gr.Image(interactive=False, label='文献表格')
            
            col_map_df = gr.DataFrame(
                value=dict2dataframe(init_colmap),
                headers=["表格字段", "数据库字段"],
                datatype=["str", "str"],
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                tabel_col_dropdown = gr.Dropdown(choices=[], label="文献表格字段", interactive=True)
                csv_col_dropdown = gr.Dropdown(choices=[], label="数据库字段", interactive=True)
            with gr.Row():
                confirm_button = gr.Button("确认修改")
                save_button = gr.Button("保存", elem_classes=["save-button"])
            clean_tag = gr.Checkbox(label="清空历史数据记录")
            with gr.Row():
                last_table = gr.Button("◀ 上一个表格", visible=True)
                input_table_idx = gr.Number(container=False, interactive=True, precision=0, minimum=1, scale=0, min_width=100)
                idx_info = gr.Markdown("### 0 / 0")
                go_btn = gr.Button("跳转", visible=True, scale=0, min_width=100)
                next_table = gr.Button("下一个表格 ▶", visible=True)
            filename = gr.Markdown("当前文件: _未上传_")
            human_download_file = gr.File(label='下载文件(修改)', visible=False)

        with gr.Column(scale=0, min_width=100):
            gr.HTML("""
                <div style="
                    border-left: 2px solid #ccc;
                    height: 100%;
                    margin: 0 10px;
                "></div>""")
                
    metedata = gr.State([])
    total_table = gr.State(value=0)
    curr_table_idx = gr.State(value=-1)
    all_csv_col = gr.State(pipeline.all_csv_col)
    colmap = gr.State(init_colmap)
    modify_tag = gr.State(False)

    def process_inputs(pdf_path):
        config = pipeline.config        
        excel_path = config.get('excel_path', '')
        csv_dir = pipeline.config.get('csv_dir', '')
        excel_to_csv(excel_path, csv_dir)

        pdf_info = pipeline(pdf_path)
        if isinstance(pdf_info, str): return pdf_info, None, 0
        
        metedata = []
        for pdf in pdf_info:
            metapath = pdf['metapath']
            with jsonlines.open(metapath, 'r') as reader:
                for item in reader: 
                    if item['type'] != 'table': continue
                    if not item['table_parse_tag']: continue
                    df_clean = pd.read_csv(item['df_clean_path'], low_memory=False)
                    item['dataframe'] = df_clean
                    metedata.append(item)
        output_excel_path = config.get('output_excel_path', '')
        csvs_to_excel(csv_dir, output_excel_path)

        return '所有PDF处理完成！', metedata, len(metedata), output_excel_path
    
    submit_button.click(process_inputs, inputs=[pdf_path], outputs=[status_message, metedata, total_table, download_file])


    def update_display(page, metedata, all_csv_col: dict):
        total = len(metedata)
        if not total:
            return {'Null':'Null'}, dict2dataframe({'Null':'Null'}), [], [], "### 0 / 0", gr.Image()
        page = max(0, min(page, total))

        item = metedata[page] if total else {}
        csv_col = all_csv_col.keys() if total else []
        csv_col = [x for x in csv_col if x not in ['编号', '资料来源/参考文献', '修改时间']] if 'table_col' in item.keys() else []
        if csv_col: csv_col.append('no_match')
        table_col = item['table_col'] if 'table_col' in item.keys() else []

        colmap = item["col_map"] if 'table_col' in item.keys() else {'Null':'Null'}
        col_map_df = dict2dataframe(colmap)
        table_col = gr.Dropdown(choices=table_col)
        csv_col = gr.Dropdown(choices=csv_col)
        idx_info = f"### {page+1} / {total}" if table_col else "### 0 / 0"
        img_path = item["img_path"] if 'img_path' in item.keys() else gr.Image()
        pdf_name = f'当前文件: _{item["pdf_name"]}_' if 'pdf_name' in item.keys() else "当前文件: _未上传_"

        return colmap, col_map_df, table_col, csv_col, idx_info, img_path, pdf_name #, df_table
    
    def last_table_fn(table_idx, total):
        if not total: return 0
        if table_idx == -1: return 0
        return max(0, table_idx - 1)
    last_table.click(
        last_table_fn, inputs=[curr_table_idx, total_table], outputs=curr_table_idx
        ).then(
            update_display,
            inputs=[curr_table_idx, metedata, all_csv_col],
            outputs=[colmap, col_map_df, tabel_col_dropdown, csv_col_dropdown, idx_info, table_image, filename]#, df_table]
        )
    def next_table_fn(table_idx, total):
        if not total: return 0
        if table_idx == -1: return 0
        return min(table_idx + 1, total - 1)
    next_table.click(
        fn=next_table_fn, inputs=[curr_table_idx, total_table], outputs=curr_table_idx
        ).then(
            fn=update_display, 
            inputs=[curr_table_idx, metedata, all_csv_col], 
            outputs=[colmap, col_map_df, tabel_col_dropdown, csv_col_dropdown, idx_info, table_image, filename]#, df_table]
        )
    def jump_page(input_table_idx, table_idx, total):
        try:
            p = int(input_table_idx)
            if 1 <= p <= total:
                return p - 1
        except:
            pass
        return table_idx
    go_btn.click(jump_page, [input_table_idx, curr_table_idx, total_table], curr_table_idx).then(
        update_display, 
        inputs=[curr_table_idx, metedata, all_csv_col], 
        outputs=[colmap, col_map_df, tabel_col_dropdown, csv_col_dropdown, idx_info, table_image, filename]
    )
    
    def confirm(tabel_col: str, csv_col: str, colmap: dict, metedata: list, table_idx: int):
        if not tabel_col or not csv_col: 
            print("警告: 存在未选择列标签!")
            col_map_df = dict2dataframe(colmap)
            return col_map_df, colmap, metedata, True
        colmap[tabel_col] = csv_col
        col_map_df = dict2dataframe(colmap)
        metedata[table_idx]['col_map'] = colmap
        metedata[table_idx]['modify'] = True
        modify_tag = True
        return (col_map_df, colmap, metedata, modify_tag)
    
    confirm_button.click(
        fn=confirm, 
        inputs=[tabel_col_dropdown, csv_col_dropdown, colmap, metedata, curr_table_idx], 
        outputs=[col_map_df, colmap, metedata, modify_tag]
    )

    
    def save_data(metedata, clean_tag, modify_tag):
        '''修改存入数据、字段匹配库'''
        if not modify_tag: return None

        config = pipeline.config
        excel_path = config.get('excel_path', '')
        csv_dir = config.get('human_csv_dir', '')

        if clean_tag or os.path.exists(csv_dir):
            excel_to_csv(excel_path, csv_dir)

        all_csv_data = {}
        for csv_file in os.listdir(csv_dir):
            csv_path = os.path.join(csv_dir, csv_file)
            df_csv = pd.read_csv(csv_path, nrows=0)
            all_csv_data[csv_path] = df_csv
        
        for item in metedata:
            if 'modify' in item.keys() and item['modify']:
                data_fill_human_modify(item['dataframe'], all_csv_data, item, config['data_fill'])
        
        output_excel_path = config.get('human_output_excel_path', '')
        csvs_to_excel(csv_dir, output_excel_path)
        return output_excel_path
    
    save_button.click(
        save_data, inputs=[metedata, clean_tag, modify_tag], outputs=[human_download_file]
    ).then(
        fn=lambda: gr.update(visible=True), 
        inputs=None,
        outputs=human_download_file
    )

    
    def clean():
        if 'cuda' in pipeline.device.type:
            torch.cuda.empty_cache()

        return gr.File(value=None), gr.File(value=None), '', gr.Checkbox(), gr.Image(value=None),\
            dict2dataframe(init_colmap), [], [], "### 0 / 0", "当前文件: _未上传_",\
            [], 0, 0, init_colmap, False
    clean_button.click(
        fn=clean,
        inputs=None,
        outputs=[
            pdf_path, download_file, status_message, clean_tag, table_image,
            col_map_df, tabel_col_dropdown, csv_col_dropdown, idx_info, filename,
            metedata, total_table, curr_table_idx, colmap, modify_tag,
        ]
    )



if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        show_error=True,      
        quiet=False,          
        inbrowser=True        
    )