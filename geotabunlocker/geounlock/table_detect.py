import torch
import numpy as np
import json
from doclayout_yolo import YOLOv10
from PIL.Image import Image
from typing import List, Dict, Any


class TableDetector:
    
    def __init__(self, config: Dict[str, Any], device):
        print("正在初始化 doclayout_yolo 模型 ...")
        assert config['weights_path']
        self.config = config
        self.device = device
        self.imgsz = config.get('imgsz', 1024)
        self.conf = config.get('confidence_threshold', 0.3)
        
        self.model = YOLOv10(model=config['weights_path']).to(device)

        print(f"doclayout_yolo 权重: {config['weights_path']}")
        print(f"doclayout_yolo 预设图像大小: {self.imgsz}")
        print(f"doclayout_yolo 置信度: {self.conf}")


    def predict(self, page_images: List[Image]) -> List[Dict[str, Any]]:
        np_imgs = [np.array(img) for img in page_images]
        det_results = self.model.predict(
            np_imgs,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device
        )

        all_detections = []
        for page_idx, page in enumerate(det_results):
            page_detections_json = json.loads(page.tojson())
            
            for detection in page_detections_json:
                if detection["name"] in ["title", "table", "table_caption", "table_footnote"]:
                    if detection["name"] == "title" and page_idx > 0:
                        continue
                    box = detection["box"]
                    all_detections.append({
                        "page_idx": page_idx,
                        "type": detection["name"],
                        "box": [box["x1"], box["y1"], box["x2"], box["y2"]],
                        "confidence": detection["confidence"]
                    })
                    
        print(f"DocLayout 推理完成，共检测到 {len(all_detections)} 个目标。")
        return all_detections
    


