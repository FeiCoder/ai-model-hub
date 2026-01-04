import openai
from typing import List, Union, Optional, Any
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.exceptions import ConnectionError, Timeout
import os

class LocalEmbeddingClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8050/v1",
        api_key: str = "EMPTY",
        model_name: str = "bge-base-zh",
        timeout: float = 30,
        max_retries: int = 3
    ):
        """
        初始化本地嵌入模型客户端

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
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((ConnectionError, Timeout))
    )
    def embed(
        self,
        input: Union[str, List[str]],
        as_array: bool = False
    ) -> Union[List[List[float]], np.ndarray]:
        """
        生成文本嵌入向量

        Parameters:
            input (Union[str, List[str]]): 输入文本或文本列表
            as_array (bool): 是否返回 numpy 数组，默认返回 list

        Returns:
            Union[List[List[float]], np.ndarray]: 嵌入向量列表或数组
        """
        if isinstance(input, str):
            input = [input]

        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings) if as_array else embeddings
    
    def embed_qwen_service(
        self,
        input: Union[str, List[str]],
        as_array: bool = False
    ) -> Union[List[List[float]], np.ndarray]:
        """
        使用Qwen云服务生成文本嵌入向量

        Parameters:
            input (Union[str, List[str]]): 输入文本或文本列表
            as_array (bool): 是否返回 numpy 数组，默认返回 list

        Returns:
            Union[List[List[float]], np.ndarray]: 嵌入向量列表或数组
        """
        # 初始化Qwen云服务客户端
        qwen_client = openai.OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 处理输入参数，确保是列表格式
        if isinstance(input, str):
            input = [input]
        
        # 调用Qwen云服务
        response = qwen_client.embeddings.create(
            model="text-embedding-v4",
            input=input,
            dimensions=1024,
            encoding_format="float"
        )
        
        # 提取嵌入向量
        embeddings = [item.embedding for item in response.data]
        
        # 根据as_array参数决定返回格式
        return np.array(embeddings) if as_array else embeddings
    
if __name__ == "__main__":
    client = LocalEmbeddingClient()
    texts = ["猪肉", "五花肉", "瘦肉", "鸡肉"]
    embeddings = client.embed(texts, as_array=True)
    # embeddings = client.embed_qwen_service(texts, as_array=True)
    print("Embeddings shape:", embeddings.shape)
    print(embeddings)