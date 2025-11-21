#!/bin/bash
# 启动并等待第一个任务完成
echo "启动第一个任务..."
nohup python Expriment_monai.py > log6_datasets2.txt 2>&1 &
first_pid=$!
tail -f log6_datasets2.txt &
tail_pid=$!
wait $first_pid
kill $tail_pid
echo "第一个任务已完成"

# 启动第二个任务
echo "启动第二个任务..."
nohup python Experiment_head.py > log7_head.txt 2>&1 &
tail -f log7_head.txt
