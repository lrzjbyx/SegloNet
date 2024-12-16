# SegLoNet

## 思路

| 对齐前 | 对齐后 | 
| ---- | ---- | 
| ![对齐前](https://github.com/lrzjbyx/align/raw/main/image.png)  |![对齐后](https://github.com/lrzjbyx/align/raw/main/result0.png) |
|  -  |![对齐后](https://github.com/lrzjbyx/align/raw/main/result.png) |



## 目录结构
```
    core            核心代码
        |
        |---        align       对齐策略
        |---        convert     标注方式转换
        |---        dataset     数据集处理
        |---        loss        损失函数
        |---        models      模型
        |---        tools       工具类
        |---        train_utils 训练脚本
    experiment      测试图片
    seal_dataset    数据集
    predictions_kernal-2-key-10.json    核大小尺寸
```

## 使用

* 安装
    ```
    pip install -r requirements.txt
    ```

* 训练
    ```
    train.py
    ```

* 评估
    ```
    eval2.py
    ```
* 预测
    ```
    predict_folder.py
    ```

## 相关
* 标注工具
    * [annotation](https://github.com/lrzjbyx/annotation)
* align
    * [CUDA align](https://github.com/lrzjbyx/align)
    * [TPS](https://github.com/lrzjbyx/TPS_CPP)