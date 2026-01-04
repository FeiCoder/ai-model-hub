if [ ! -d logs ]; then
  mkdir logs
fi
# 设置gpu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
# 启动服务
nohup python -u app/embedding/main.py > logs/emb_api.log 2>&1 &
nohup python -u app/reranker/main.py > logs/reranker_api.log 2>&1 &