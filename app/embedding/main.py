import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import uvicorn
from app.core.config import load_config
from app.core.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingObject

# 加载配置
config = load_config()

# 创建FastAPI应用
app = FastAPI(title="Local Embedding API", version="1.0.0")

# 加载所有配置的模型
loaded_models: Dict[str, SentenceTransformer] = {}
default_model_name = None

print("Loading embedding models...")
for model_conf in config.get("embedding_models", []):
    name = model_conf["name"]
    path = model_conf["path"]
    device = model_conf.get("device", "cuda")
    
    # 优先使用环境变量覆盖路径
    if os.getenv("EMBEDDING_MODEL_PATH") and model_conf.get("default", False):
        path = os.getenv("EMBEDDING_MODEL_PATH")
        
    try:
        print(f"Loading model: {name} from {path} on {device}")
        model = SentenceTransformer(path, device=device)
        loaded_models[name] = model
        
        if model_conf.get("default", False):
            default_model_name = name
    except Exception as e:
        print(f"Failed to load model {name}: {str(e)}")

if not loaded_models:
    raise RuntimeError("No embedding models loaded successfully")

if not default_model_name:
    default_model_name = list(loaded_models.keys())[0]

print(f"Models loaded: {list(loaded_models.keys())}, Default: {default_model_name}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    生成文本嵌入向量
    """
    try:
        # 确定使用的模型
        model_name = request.model if request.model else default_model_name
        
        if model_name not in loaded_models:
            if not request.model:
                 model_name = default_model_name
            else:
                 raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Available: {list(loaded_models.keys())}")
        
        model = loaded_models[model_name]

        # 处理输入，支持单个字符串或列表
        input_texts = request.input
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # 记录时间
        start_time = time.time()
        # 生成嵌入
        embeddings = model.encode(
            input_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        duration = time.time() - start_time
        print(f"Embedding generation time: {duration:.2f} seconds")
        
        # 构建响应
        data = [
            EmbeddingObject(
                object="embedding",
                embedding=embedding.tolist(),
                index=i
            )
            for i, embedding in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage={"total_tokens": sum(len(t) for t in input_texts), "prompt_tokens": 0, "total_time": duration} # 简单估算
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动API服务
if __name__ == "__main__":
    port = int(os.getenv("PORT", config["server"]["embedding_port"]))
    host = config["server"]["host"]
    uvicorn.run("app.embedding.main:app", host=host, port=port, workers=config["embedding_models"][0].get("workers", 1))
