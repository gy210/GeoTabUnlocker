import gradio as gr
import yaml
import os
from collections.abc import MutableMapping

CONFIG_FILE = './configs/admin.yaml'

def load_config():
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            'device': 'cpu',
            'api_key': 'YOUR_API_KEY_HERE',
            'pdf_processing': {'dpi': 300},
            'table_detector': {'imgsz': 1024, 'confidence_threshold': 0.4},
            'data_fill': {'llm': {'max_retry': 2, 'llm_model': 'glm-4-plus'}}
        }
        save_config(default_config)
        return default_config
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config_data):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, sort_keys=False)


def update_and_save_config(device, api_key, dpi, imgsz, confidence, max_retry, llm_model):
    config = load_config()

    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, MutableMapping):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    updates = {
        'device': device,
        'api_key': api_key,
        'pdf_processing': {'dpi': dpi},
        'table_detector': {'imgsz': imgsz, 'confidence_threshold': confidence},
        'data_fill': {'llm': {'max_retry': max_retry, 'llm_model': llm_model}}
    }
    
    updated_config = update_nested_dict(config, updates)
    
    save_config(updated_config)
    
    return f"Save！"

initial_config = load_config()

def get_nested_value(keys, default=None):
    d = initial_config
    for key in keys:
        d = d.get(key, {})
    return d if not isinstance(d, dict) else default

def exit_app():
    os._exit(0)


with gr.Blocks(title="系统配置编辑器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 系统参数配置")
    gr.Markdown("在此界面中修改系统参数，点击下方的“保存配置”按钮以生效。")

    with gr.Tabs():
        with gr.TabItem("通用设置"):
            device_radio = gr.Radio(
                ["cpu", "cuda:0"],
                value=get_nested_value(['device'], 'cpu'),
                label="运行设备 (device)",
                info="选择用于计算的设备"
            )
            api_key_textbox = gr.Textbox(
                value=get_nested_value(['api_key'], ''),
                label="API Key",
                info="您的 API 密钥"
            )
            dpi_number = gr.Number(
                value=get_nested_value(['pdf_processing', 'dpi'], 300),
                label="PDF 处理 DPI (dpi)",
                precision=0,
                info="渲染 PDF 时的图像分辨率"
            )

        with gr.TabItem("模型参数"):
            imgsz_slider = gr.Slider(
                minimum=512,
                maximum=2048,
                step=64,
                value=get_nested_value(['table_detector', 'imgsz'], 1024),
                label="表格检测图像尺寸 (imgsz)",
                info="输入给表格检测模型的图像大小"
            )
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=get_nested_value(['table_detector', 'confidence_threshold'], 0.4),
                label="表格检测置信度阈值",
                info="只有高于此阈值的检测结果才会被接受"
            )
            max_retry_number = gr.Number(
                value=get_nested_value(['data_fill', 'llm', 'max_retry'], 2),
                label="LLM 最大重试次数",
                precision=0
            )
            llm_model_dropdown = gr.Dropdown(
                ["glm-4-plus", "glm-4", "glm-3-turbo", "gpt-4-turbo", "gpt-3.5-turbo"],
                value=get_nested_value(['data_fill', 'llm', 'llm_model'], 'glm-4-plus'),
                label="大语言模型 (llm_model)",
                allow_custom_value=True,
                info="选择用于数据填充的大语言模型，也支持手动输入"
            )

    status_output = gr.Textbox(label="状态", interactive=False)
    with gr.Row():
        save_button = gr.Button("保存配置", variant="primary")
        exit_button = gr.Button("退出应用", variant="stop")
    
    save_button.click(
        fn=update_and_save_config,
        inputs=[
            device_radio, 
            api_key_textbox, 
            dpi_number, 
            imgsz_slider, 
            confidence_slider, 
            max_retry_number, 
            llm_model_dropdown
        ],
        outputs=status_output
    )
    exit_button.click(
        fn=exit_app,
    )

if __name__ == "__main__":
    demo.launch()