# AI Model Hub

本项目提供了一个基于 GPU 的本地模型服务中心，专注于为智能体提供基础模型能力，包括文本嵌入（Embedding）和重排序（Reranker）。

项目采用 `uv` 进行 Python 依赖管理，并支持 Docker 容器化部署。

## 功能特性

- **Embedding Service**: 提供文本向量化服务，兼容 OpenAI API 格式。
  - 默认模型: `bge-base-zh-v1.5`
  - 端口: 8050
- **Reranker Service**: 提供文档重排序服务。
  - 默认模型: `bge-reranker-v2-m3`
  - 端口: 8051
- **高性能**: 基于 FastAPI 和 Uvicorn，支持 GPU 加速。
- **易部署**: 提供 Docker Compose 一键部署方案。
- **灵活配置**: 支持通过 YAML 配置文件管理多个模型。

## 目录结构

```
ai-model-hub/
├── app/                # 应用源代码
│   ├── core/           # 核心模块 (配置, 数据模型)
│   ├── embedding/      # Embedding 服务
│   └── reranker/       # Reranker 服务
├── config/             # 配置文件
│   └── config.yaml
├── clients/            # 客户端示例代码
├── scripts/            # 辅助脚本
├── models/             # 模型文件存放目录（需自行下载）
├── Dockerfile          # Docker 构建文件
├── docker-compose.yml  # Docker Compose 编排文件
├── pyproject.toml      # uv 依赖配置
└── README.md           # 说明文档
```

## 部署指南

### 1. 环境准备

- Linux 操作系统
- NVIDIA GPU 及驱动
- Docker & Docker Compose
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (用于 Docker 调用 GPU)

### 2. 准备模型

在项目根目录下创建 `models` 文件夹，并下载所需的模型文件。

```bash
mkdir models
cd models

# 示例：使用 git lfs 下载模型 (需安装 git-lfs)
# 下载 Embedding 模型
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5

# 下载 Reranker 模型
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3
```

确保目录结构如下：
```
models/
├── bge-base-zh-v1.5/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── bge-reranker-v2-m3/
    ├── config.json
    ├── model.safetensors
    └── ...
```

### 3. 配置模型

修改 `config/config.yaml` 文件以配置需要加载的模型：

```yaml
embedding_models:
  - name: "bge-base-zh"
    path: "/models/bge-base-zh-v1.5"
    device: "cuda"
    default: true
  # 可以添加更多模型
  # - name: "bge-m3"
  #   path: "/models/bge-m3"
  #   device: "cuda"

reranker_models:
  - name: "bge-reranker-v2-m3"
    path: "/models/bge-reranker-v2-m3"
    device: "cuda"
    default: true
```

### 4. 启动服务

使用 Docker Compose 启动服务：

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

### 5. 验证服务

**Embedding API 测试:**

```bash
curl -X POST http://localhost:8050/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["测试文本1", "测试文本2"],
    "model": "bge-base-zh"
  }'
```

**Reranker API 测试:**

```bash
curl -X POST http://localhost:8051/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "passages": ["人工智能是计算机科学的一个分支", "今天天气不错"],
    "model": "bge-reranker-v2-m3"
  }'
```

## 本地开发

如果你想在本地直接运行代码（非 Docker）：

1.  安装 `uv`:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  创建虚拟环境并安装依赖:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install .
    ```

3.  启动:
    ```bash
    sh ./scripts/init.sh
    ```
4.  关闭：
    ```bash
    sh ./scripts/shutdown.sh
    ```

## 客户端使用

项目提供了 Python 客户端示例代码：

- `clients/embedding_client.py`: Embedding 服务客户端
- `clients/reranker_client.py`: Reranker 服务客户端

可以在你的应用中直接引用这些客户端类来调用服务。
