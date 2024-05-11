scripts:
    nohup python main.py >output.log 2>&1 &
    tensorboard --logdir=./result/summary --port 6006
请注意推理加载的模型需要和当前模型是同一个
使用多进程读取数据的话需要修改comment
目录结构：
    project
        dataset
            images
            train.csv
            test.csv
        result
            output_log
            predicts
            summary
        saved_models
        
