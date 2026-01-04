import os
import openai
from typing import Dict, List, Union, Optional
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.exceptions import ConnectionError, Timeout

class LocalRerankerClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8051/v1",
        api_key: str = "EMPTY",
        model_name: str = "bge-reranker-v2-m3",
        timeout: float = 30,
        max_retries: int = 3
    ):
        """
        初始化本地重排序模型客户端

        Parameters:
            base_url (str): 服务地址
            api_key (str): API 密钥（本地服务通常设为 "EMPTY"）
            model_name (str): 模型名称
            timeout (float): 请求超时时间（秒）
            max_retries (int): 最大重试次数
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        # 注意：这里我们不使用openai库，而是使用requests直接调用
        import requests
        self.session = requests.Session()
        self.session.timeout = self.timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((ConnectionError, Timeout))
    )
    def rerank(
        self,
        query: str,
        passages: List[str],
        return_documents: bool = True
    ) -> List[dict]:
        """
        对查询和文档进行重排序

        Parameters:
            query (str): 查询文本
            passages (List[str]): 文档列表
            return_documents (bool): 是否返回原始文档内容

        Returns:
            List[dict]: 重排序结果，包含索引、相关性分数和可选的文档内容
        """
        import requests
        
        payload = {
            "query": query,
            "passages": passages,
            "model": self.model_name,
            "return_documents": return_documents
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            result = response.json()
            return result["data"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to reranker service: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error during reranking: {str(e)}")

    def rerank_qwen_service(self, query: str, passages: List[str]):
        """
        使用 Qwen 服务进行重排序

        Parameters:
            query (str): 查询文本
            passages (List[str]): 文档列表

        Returns:
            List[dict]: 重排序结果，包含索引、相关性分数和文档内容
        """
        import dashscope
        from http import HTTPStatus
        
        # 设置API密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")
        
        try:
            # 调用DashScope的rerank接口
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=passages,
                top_n=len(passages),  # 返回所有文档的排序结果
                return_documents=True,
                api_key=api_key
            )
            
            # 检查响应状态
            if response.status_code == HTTPStatus.OK:
                # 转换结果格式以匹配原有返回格式
                results = []
                for item in response.output['results']:
                    results.append({
                        'index': item['index'],
                        'relevance_score': item['relevance_score'],
                        'document': item['document']
                    })
                return results
            else:
                raise RuntimeError(f"Qwen reranking failed with status {response.status_code}: {response.message}")
                
        except Exception as e:
            raise RuntimeError(f"Error during Qwen reranking: {str(e)}")

if __name__ == "__main__":
    client = LocalRerankerClient()
    query = "鲜筒骨"
    passages = [
        "蛏子肉",
        "五花肉是猪肉的一种，肥瘦相间，口感鲜美",
        "瘦肉富含蛋白质，脂肪含量较低",
        "猪肘棒"
    ]
    
    results = client.rerank(query, passages)
    print("Rerank Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['relevance_score']:.4f}, Document: {result['document']}")