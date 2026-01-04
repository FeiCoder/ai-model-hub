# 使用 NVIDIA CUDA 基础镜像以支持 GPU
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# === 关键优化 1：换 APT 源为阿里云（国内加速）===
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# 安装 Python 和必要的系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 从官方镜像复制 uv (最稳定可靠的方式)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 验证安装
RUN uv --version

# 设置工作目录
WORKDIR /app

# 创建虚拟环境
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# === 关键优化 3：先复制 pyproject.toml，单独安装依赖（利用缓存）===
COPY pyproject.toml .

# 安装依赖（使用国内 PyPI 镜像加速，可选）
# 注意：torch 仍需官方 cu121 源，其他包可用清华源
RUN uv pip install . \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制源代码（放最后，避免因代码变更导致重装依赖）
COPY app/ ./app/
COPY config/ ./config/

# 设置 PYTHONPATH
ENV PYTHONPATH=/app

# 默认命令
CMD ["python", "app/embedding/main.py"]