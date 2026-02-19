import torch
import logging
logging.basicConfig(format='%(asctime)s  | INFO    |    %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return f'cuda:{i}'
    return 'cpu'


def check_api_key(api_key: str) -> None:
    if not isinstance(api_key, str):
        raise ValueError("API Key 必须是字符串类型！")
    if len(api_key) != 49:
        raise ValueError("API Key 长度错误，必须为 49 个字符！")
    if '.' not in api_key:
        raise ValueError("API Key 必须包含分隔符 '.'！")
    
    prefix, suffix = api_key.split('.')
    if len(prefix) != 32:
        raise ValueError("API Key 的前缀部分必须为 32 个字符！")
    if not suffix:
        raise ValueError("API Key 的后缀部分不能为空！")

    logging.info(f"api_key正确!")