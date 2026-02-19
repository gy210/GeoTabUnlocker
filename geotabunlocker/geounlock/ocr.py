import torch
import numpy as np
from paddleocr import PaddleOCR
from PIL.Image import Image
from typing import List, Dict, Any

class OcrExtractor:

    def __init__(self, config: Dict[str, Any], device: torch.device):
        print("正在初始化 PaddleOCR 引擎 (这可能需要一些时间)...")

        self.config = config
        self.device = device
        
        lang = config.get('lang', 'ch')
        use_textline_orientation = config.get('use_textline_orientation', False)

        self.ocr_engine = PaddleOCR(
            use_textline_orientation=use_textline_orientation,
            lang=lang, 
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

        print("PaddleOCR 引擎初始化完毕。")


    def predict(self, image: Image) -> List[Any]:
        image_np = np.array(image)
        result = self.ocr_engine.predict(image_np, cls=True)
        if result and result[0] is not None:
            return result[0]
        return []
    

    def predict_batch(self, images: List[Image]) -> List[List[Any]]:
        if not images:
            return []
        if not isinstance(images, list):
            return []

        image_np_list = [np.array(img) for img in images]
        batch_results = self.ocr_engine.predict(image_np_list)

        outputs = []
        for res_data in batch_results:
            cleaned_item = {}
            if res_data:
                cleaned_item["rec_texts"] = res_data.get('rec_texts', [])
                cleaned_item["rec_scores"] = res_data.get('rec_scores', np.array([], dtype=np.float32))
                cleaned_item["rec_polys"] = res_data.get('rec_polys', np.array([], dtype=np.int16))
                cleaned_item["rec_boxes"] = res_data.get('rec_boxes', np.array([], dtype=np.int16))
            
            outputs.append(cleaned_item)

        return outputs
