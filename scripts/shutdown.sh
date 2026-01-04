#!/bin/bash
# model_stop.sh (增强版)

echo "正在停止模型服务..."

# 函数：停止指定进程
stop_process() {
    local process_name=$1
    local process_identifier=$2
    
    echo "查找 $process_name 进程..."
    pids=$(ps aux | grep "$process_identifier" | grep -v grep | awk '{print $2}')
    
    if [ ! -z "$pids" ]; then
        for pid in $pids; do
            echo "停止 $process_name (PID: $pid)"
            kill $pid
            
            # 等待进程结束，最多等待10秒
            local count=0
            while ps -p $pid > /dev/null 2>&1; do
                if [ $count -ge 10 ]; then
                    echo "进程未正常停止，强制终止 PID: $pid"
                    kill -9 $pid
                    break
                fi
                sleep 1
                ((count++))
            done
        done
    else
        echo "未找到运行中的 $process_name 服务"
    fi
}

# 停止 embedding API 服务
stop_process "Embedding API" "app/embedding/main.py"

# 停止 reranker API 服务
stop_process "Reranker API" "app/reranker/main.py"

echo "模型服务停止操作完成"