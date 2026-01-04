import os
import time
import torch
from fastapi import FastAPI, HTTPException
from typing import List, Union, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn
import torch.nn.functional as F
from app.core.config import load_config
from app.core.schemas import RerankRequest, RerankResponse, RerankResult

# 加载配置
config = load_config()

# 创建FastAPI应用
app = FastAPI(title="Local Reranker API", version="1.0.0")

# 加载所有配置的模型
loaded_models: Dict[str, dict] = {} # {"name": {"model": model, "tokenizer": tokenizer}}
default_model_name = None

print("Loading reranker models...")
for model_conf in config.get("reranker_models", []):
    name = model_conf["name"]
    path = model_conf["path"]
    device = model_conf.get("device", "cuda")
    
    # 优先使用环境变量覆盖路径
    if os.getenv("RERANKER_MODEL_PATH") and model_conf.get("default", False):
        path = os.getenv("RERANKER_MODEL_PATH")
        
    try:
        print(f"Loading model: {name} from {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            print(f"Model {name} loaded on GPU")
        else:
            print(f"Model {name} loaded on CPU")
            
        model.eval()
        loaded_models[name] = {"model": model, "tokenizer": tokenizer, "device": device}
        
        if model_conf.get("default", False):
            default_model_name = name
    except Exception as e:
        print(f"Failed to load model {name}: {str(e)}")

if not loaded_models:
    raise RuntimeError("No reranker models loaded successfully")

if not default_model_name:
    default_model_name = list(loaded_models.keys())[0]

print(f"Models loaded: {list(loaded_models.keys())}, Default: {default_model_name}")

@app.post("/v1/rerank", response_model=RerankResponse)
async def create_rerank(request: RerankRequest):
    """
    对查询和文档对进行重排序
    """
    try:
        # 确定使用的模型
        model_name = request.model if request.model else default_model_name
        
        if model_name not in loaded_models:
             if not request.model:
                 model_name = default_model_name
             else:
                 raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Available: {list(loaded_models.keys())}")
        
        model_instance = loaded_models[model_name]["model"]
        tokenizer_instance = loaded_models[model_name]["tokenizer"]
        device = loaded_models[model_name]["device"]

        # 记录时间
        start_time = time.time()
        # 构建查询-文档对
        pairs = [[request.query, passage] for passage in request.passages]
        
        # 修改推理部分
        with torch.no_grad():
            inputs = tokenizer_instance(pairs, padding=True, truncation=True, 
                            return_tensors='pt', max_length=512)
            
            # 如果使用GPU，将输入数据移到GPU
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            scores = model_instance(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = F.sigmoid(scores)  # 转换为概率分数
        
        # 构建结果
        scores_list = scores.tolist()
        sorted_results = sorted(enumerate(scores_list), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_results:
            result = RerankResult(
                index=idx,
                relevance_score=score
            )
            if request.return_documents:
                result.document = request.passages[idx]
            results.append(result)
            
        duration = time.time() - start_time
        print(f"Reranking time: {duration:.2f} seconds")
        
        return RerankResponse(
            data=results,
            model=model_name,
            usage={"total_docs": len(request.passages), "total_time": duration}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动API服务
if __name__ == "__main__":
    port = int(os.getenv("PORT", config["server"]["reranker_port"]))
    host = config["server"]["host"]
    uvicorn.run(app, host=host, port=port, workers=config["reranker_models"][0].get("workers", 1))
