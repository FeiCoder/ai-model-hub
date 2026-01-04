import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    优先使用传入的路径，其次查找环境变量 CONFIG_PATH，最后默认使用 config/config.yaml
    """
    if config_path is None:
        # 默认路径调整为相对于项目根目录的 config/config.yaml
        # 假设当前文件在 app/core/config.py，项目根目录在 ../../
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(base_dir, "config", "config.yaml")
        config_path = os.getenv("CONFIG_PATH", default_path)
    
    if not os.path.exists(config_path):
        # 尝试在当前工作目录查找
        if os.path.exists("config/config.yaml"):
            config_path = "config/config.yaml"
        else:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
