from PIL import Image, ImageOps

import os
from glob import glob
from pathlib import Path
import json
import jsonlines
import torch
from typing import Dict, Any, List
from ruamel.yaml import YAML
from copy import deepcopy

import pandas as pd
import numpy as np


from geounlock.table_detect import TableDetector
from geounlock.ocr import OcrExtractor
from geounlock.tsr_pipeline import TaBLIP_pipeline
from geounlock.utils import (
    convert_pdf_to_images,
    clean_and_convert_data_robust,
)
from geounlock.html_converter import html_table_to_excel
from geounlock.table_header_parser import header_parser
from geounlock.fill_in_database import data_fill
from geounlock.literature_source import determine_title_with_llm
from geounlock.table_title import extract_location_from_title


class DocumentExtractionPipeline:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.ocr_device = 'gpu' if 'cuda' in config.get('device', 'cpu') else 'cpu'
        self.api_key = config['api_key']

        # model
        self.tr_detector = TableDetector(config['table_detector'], self.device)
        self.ocr_extractor = OcrExtractor(config['ocr_extractor'], self.ocr_device)
        self.tablip = TaBLIP_pipeline(config['tablip'], self.device)
        
        self.all_csv_data = {}
        self.all_csv_col = {}
        self.get_all_csv_data()
    
    def get_all_csv_data(self):
        csv_dir = self.config.get('csv_dir', '')
        all_csv_data = {}
        all_csv_col = {}
        for csv_file in os.listdir(csv_dir):
            csv_path = os.path.join(csv_dir, csv_file)
            
            df_csv = pd.read_csv(csv_path, nrows=0)
            all_csv_data[csv_path] = df_csv

            for csv_col in df_csv.columns:
                if csv_col not in all_csv_col.keys():
                    all_csv_col[csv_col] = []
                all_csv_col[csv_col].append(csv_path)

        self.all_csv_data = all_csv_data
        self.all_csv_col = all_csv_col


    def process_pdf(self, pdf_path: Path, auto: bool = True):
        api_key = self.api_key

        pdf_name = pdf_path.stem
        output_dir = Path(self.config['paths']['output_dir']) / pdf_name
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        page_images = convert_pdf_to_images(
            pdf_path, 
            dpi=self.config['pdf_processing']['dpi'])

        metadata = []
        image_items = []
        table_caption = {}
        table_footnote = {}
        threshold = 100

        tr_outputs = self.tr_detector.predict(page_images)
        org_index = 0
        for i, detection in enumerate(tr_outputs):
            page_idx = detection["page_idx"]
            box = detection["box"]
            
            if detection['type'] in ['table_caption', 'table_footnote']:
                if detection['type'] == 'table_caption':
                    if page_idx not in table_caption.keys():
                        table_caption[page_idx] = []
                    table_caption[page_idx].append(box)
                if detection['type'] == 'table_footnote':
                    if page_idx not in table_footnote.keys():
                        table_footnote[page_idx] = []
                    table_footnote[page_idx].append(box)
                continue

            if detection['type'] == 'title':
                W, H = page_images[page_idx].size
                x1, y1, x2, y2 = (max(0, box[0] - 100), max(0, box[1] - 50), min(W, box[2] + 100), min(H, box[3] + 220))
                image = page_images[page_idx].crop((x1, y1, x2, y2))
            else:
                image = page_images[page_idx].crop(box)
            image = ImageOps.expand(image, border=(50, 50, 50, 50), fill=(255, 255, 255))

            img_name = f"page_{page_idx}_{i}_{detection['type']}"
            img_path = os.path.join(images_dir, img_name + '.jpg')
            image.save(img_path)
            
            detection['pdf_path'] = str(pdf_path)
            detection['pdf_name'] = pdf_name
            detection['img_path'] = img_path
            detection['img_name'] = img_name
            metadata.append(detection)

            if detection['type'] in ['table', 'title']: 
                image_items.append({
                    'image': image,
                    'org_index': org_index,
                    'type': detection['type'],
                })
                org_index += 1
        
        index = -1
        for i, detection in enumerate(tr_outputs):
            if detection['type'] in ['table', 'title']: index += 1
            if detection['type'] != 'table': continue
            page_idx = detection["page_idx"]
            (x1, y1, x2, y2) = detection["box"]

            upper_y1, lower_y2 = deepcopy(y1), deepcopy(y2)
            center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if page_idx in table_caption.keys():
                for (cx1, cy1, cx2, cy2) in table_caption[page_idx]:
                    center2 = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
                    if center1[1] > center2[1] and abs(center1[0] - center2[0]) < threshold: 
                        dist = np.sqrt((center1[0] - center2[0])**2 + (cy2 - y1)**2)
                        if dist < threshold and cy1 < upper_y1: upper_y1 = cy1
            if page_idx in table_footnote.keys():    
                for (fx1, fy1, fx2, fy2) in table_footnote[page_idx]:
                    center2 = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                    if center1[1] < center2[1] and abs(center1[0] - center2[0]) < threshold: 
                        dist = np.sqrt((center1[0] - center2[0])**2 + (fy1 - y2)**2)
                        if dist < threshold and fy2 > lower_y2: lower_y2 = fy2

            if upper_y1 - y1 != 0:
                caption_img = page_images[page_idx].crop((x1, upper_y1, x2, y1))
                img_name = f"page_{page_idx}_{i}_table_caption.jpg"
                img_path = os.path.join(images_dir, img_name)
                caption_img.save(img_path)
                if index == image_items[index]['org_index']:
                    image_items[index]['table_caption'] = caption_img
                
            if lower_y2 - y2 != 0:
                footnote_img = page_images[page_idx].crop((x1, y2, x2, lower_y2))
                img_name = f"page_{page_idx}_{i}_table_footnote.jpg"
                img_path = os.path.join(images_dir, img_name)
                footnote_img.save(img_path)
                if index == image_items[index]['org_index']:
                    image_items[index]['table_footnote'] = footnote_img

        B = self.config['ocr_extractor'].get('batch_size', 16)
        for i in range(0, len(image_items), B):
            mini_batch_items = image_items[i : i + B]
            mini_batch_images = [item['image'] for item in mini_batch_items]

            ocr_outputs = self.ocr_extractor.predict_batch(mini_batch_images)

            for item, ocr_data in zip(mini_batch_items, ocr_outputs):
                index = item['org_index']

                ocr_data_copy = deepcopy(ocr_data)
                ocr_data_copy['rec_polys'] = [a.tolist() for a in ocr_data_copy['rec_polys']]
                ocr_data_copy['rec_boxes'] = [a.tolist() for a in ocr_data_copy['rec_boxes']]
                metadata[index]['ocr_data'] = ocr_data_copy
                item['ocr_data'] = ocr_data

        table_image_items = [item for item in image_items if item['type'] == 'table']
        for i in range(0, len(table_image_items), B):
            mini_batch_items = table_image_items[i : i + B]
            mini_batch_captimages = [item['table_caption'] for item in mini_batch_items if 'table_caption' in item.keys()]
            mini_batch_footimages = [item['table_footnote'] for item in mini_batch_items if 'table_footnote' in item.keys()]
            mini_batch_images = mini_batch_captimages + mini_batch_footimages
            C, F = len(mini_batch_captimages), len(mini_batch_footimages)

            ocr_outputs = self.ocr_extractor.predict_batch(mini_batch_images)

            c_i, f_i = 0, 0
            for i, item in enumerate(mini_batch_items):
                index = item['org_index']
                if 'table_caption' in item.keys() and c_i < C:
                    metadata[index]['table_caption'] = ' '.join(ocr_outputs[c_i]['rec_texts'])
                    c_i += 1
                if 'table_footnote' in item.keys():
                    metadata[index]['table_caption'] += ' ' + ' '.join(ocr_outputs[f_i+C]['rec_texts'])
                    f_i += 1

        print(f"PaddleOCR 推理完成。")

        table_index = [item['org_index'] for item in image_items if item['type'] == 'table']
        table_image = [item['image'] for item in image_items if item['type'] == 'table']
        table_ocr = [item['ocr_data'] for item in image_items if item['type'] == 'table']

        B = self.config['tablip'].get('batch_size', 16)
        for i in range(0, len(table_image), B):
            mini_batch_index = table_index[i : i + B]
            mini_batch_image = table_image[i : i + B]
            mini_batch_ocr = table_ocr[i : i + B]

            tsr_outputs = self.tablip.predict(mini_batch_image, mini_batch_ocr)
            
            for index, tsr_result in zip(mini_batch_index, tsr_outputs['html']):
                metadata[index]['html'] = tsr_result

        print(f"TaBLIP 推理完成。")

        for item in metadata:
            if item['type'] != 'table': continue

            df_orgin, html_convert = html_table_to_excel(item['html'], item, output_dir)
            item['html_convert'] = html_convert

            df_parse, flat_cols = header_parser(df_orgin, item)

            df_clean = clean_and_convert_data_robust(df_parse)

            if '.' in item['img_name']:
                img_name = item['img_name'].split('.')[0]
            else:
                img_name = item['img_name']

            if not df_clean.empty:
                df_orgin.to_csv(output_dir / f'{img_name}_orgin.csv', mode='w', index=False, encoding='utf-8-sig')
                df_parse.to_csv(output_dir / f'{img_name}_parse.csv', mode='w', index=False, encoding='utf-8-sig')
                df_clean.to_csv(output_dir / f'{img_name}_clean.csv', mode='w', index=False, encoding='utf-8-sig')
                item['dataframe'] = df_clean  
                item['df_clean_path'] = str(output_dir / f'{img_name}_clean.csv')
                item['table_parse_tag'] = True
            else:
                print(f"未保存 {item['img_name']} 数据，表格解析有误")
                item['table_parse_tag'] = False 
                item['dataframe'] = pd.DataFrame()
        
        source_may = []
        for item in metadata:
            if item['type'] != 'title': continue
            may_title = ' '.join(item['ocr_data']['rec_texts'])
            source_may.append(may_title.replace(' ', ''))
        print(f"\n可能的参考文献标题: {source_may} \n")
        source = determine_title_with_llm(source_may, api_key)
        

        if auto:
            for item in metadata:
                if item['type'] != 'table': continue
                if not item['table_parse_tag']: continue
                sample_location = ''
                if 'table_caption' in item.keys():
                    sample_location = extract_location_from_title(item['table_caption'], api_key)
                if not sample_location:
                    sample_location = extract_location_from_title(source, api_key)
                item['sample_location'] = sample_location if sample_location else ''
                item['source'] = source
                df = item['dataframe']
                all_csv_data = self.all_csv_data
                table_all_map = data_fill(df, all_csv_data, item, self.config['data_fill'], api_key)
                item['col_map'] = table_all_map if table_all_map else {}
                item['table_col'] = df.columns.tolist()

        metapath = os.path.join(output_dir, 'detection_results.jsonl')
        with jsonlines.open(metapath, 'w') as writer:
            for item in metadata:
                if item['type'] != 'table': continue
                data = deepcopy(item)
                if 'dataframe' in data:
                    del data['dataframe']
                writer.write(data)
            
        print(f"处理完成！结果已保存至: {output_dir}")
        return {
            'metapath': metapath,
            'pdf_name': pdf_name,
        }
    

    def __call__(self, path: str, auto: bool = True) -> List[dict]:

        path = Path(path)

        if not path.exists():
            print(f"错误：输入路径不存在 -> '{path}'")
            return "错误：输入路径不存在 -> '{path}'"

        pdf_paths = []
            
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_paths.append(path.resolve())
        elif path.is_dir():
            print(f"检测到输入为目录: '{path}'")

            found_files = list(path.rglob('*'))
            for file_path in found_files:
                if file_path.suffix.lower() == '.pdf' and file_path.is_file():
                    pdf_paths.append(file_path.resolve())
        else:
            print(f"错误：输入路径既不是一个有效的文件也不是目录 -> '{path}'")
            return f"错误：输入路径既不是一个有效的文件也不是目录 -> '{path}'"

        if not pdf_paths:
            print("未在指定路径中找到任何PDF文件。")
            return "未在指定路径中找到任何PDF文件。"
            
        print(f"成功找到 {len(pdf_paths)} 个PDF文件进行处理。")

        pdf_info = []
        for pdf in pdf_paths:
            output = self.process_pdf(pdf, auto)
            pdf_info.append(output)

        print(f"所有PDF文件均已处理完成!\n")

        return pdf_info



def get_pipeline():
    yaml = YAML(typ='safe') 
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.load(f)
    pipeline = DocumentExtractionPipeline(config)
    return pipeline