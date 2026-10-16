# ML_Chem
End-to-end machine learning for chemistry: from RDKit molecular representations (Morgan fingerprints, SMILES, molecular graphs) to RNNs (GRU, LSTM), CNNs (shallow and deep), and GNNs (shallow and deep) for aqueous solubility prediction

# ML_CHEM - 分子机器学习练习项目

分子表示学习与性质预测的完整实践教程。

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.03%2B-green.svg)](https://www.rdkit.org/)

## 📖 项目简介

本项目是一个面向化学信息学与分子机器学习的实践教程，涵盖了分子表示学习的核心方法。通过一系列独立可运行的Python脚本，您可以深入理解：

- 使用RDKit进行分子操作（SMILES解析、指纹计算、分子图构建）
- 分子相似性计算（Tanimoto相似度）
- 4种主流分子表示方法及对应的深度学习模型：
  - 🧬 **分子指纹 + 前馈神经网络**
  - 📝 **SMILES序列 + 循环神经网络（RNN）**
  - 📝 **SMILES序列 + 卷积神经网络（CNN）**
  - 🔗 **分子图 + 图神经网络（GNN）**

## 🚀 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd ~/ml_chem

# 安装依赖
pip install -r requirements.txt

# 安装项目（开发模式）
pip install -e .
```

### 2. 准备数据

```bash
python scripts/01_download_data.py
```

### 3. 运行所有任务

```bash
# 按顺序运行所有脚本
for script in scripts/*.py; do
    echo "运行: $script"
    python $script
    echo ""
done
```

## 📂 项目结构

```
ml_chem/
├── ml_chem/                      # 核心模块库
│   ├── __init__.py              # 包初始化
│   ├── featurizers.py           # 分子特征化器
│   ├── graphs.py                # 图数据结构
│   ├── models.py                # 神经网络模型定义
│   ├── data_utils.py            # 数据加载与预处理
│   └── train_utils.py           # 训练辅助工具
├── scripts/                      # 可执行脚本
│   ├── 01_download_data.py      # 数据下载
│   ├── 02_rdkit_basics.py       # RDKit基础操作
│   ├── 03_tanimoto_similarity.py # 任务1：Tanimoto相似度
│   ├── 04_graph_representation.py # 任务2：分子图表示
│   ├── 05_solubility_histogram.py # 任务3：溶解度分布
│   ├── 06_fingerprint_nn.py     # 任务4：指纹 + 前馈神经网络
│   ├── 07_smiles_encoding.py    # 任务5：SMILES序列编码
│   ├── 08_smiles_rnn.py         # 任务6：SMILES + RNN
│   ├── 09_smiles_cnn.py         # 任务7：SMILES + CNN
│   ├── 10_graphs_class.py       # 任务8：图数据结构
│   └── 11_graph_gnn.py          # 任务9：图神经网络
├── data/                         # 数据目录
├── tests/                        # 单元测试
├── README.md                     # 项目说明
├── requirements.txt              # 依赖列表
└── setup.py                      # 安装配置
```

## 📊 任务说明

### 前置任务：RDKit基础 (02_rdkit_basics.py)

演示如何使用RDKit读取SMILES字符串，遍历原子和键，查看分子属性。

```bash
python scripts/02_rdkit_basics.py
```

#### 运行结果示例
```bash
============================================================
RDKit基础操作演示
============================================================

扑热息痛SMILES: CC(=O)Nc1ccc(O)cc1
原子数: 11
键数: 11

原子信息:
  原子: C, 原子序数: 6, H数: 3
  原子: C, 原子序数: 6, H数: 0
  原子: O, 原子序数: 8, H数: 0
  原子: N, 原子序数: 7, H数: 1
  原子: C, 原子序数: 6, H数: 0
  原子: C, 原子序数: 6, H数: 1
  原子: C, 原子序数: 6, H数: 1
  原子: C, 原子序数: 6, H数: 0
  原子: O, 原子序数: 8, H数: 1
  原子: C, 原子序数: 6, H数: 1
  原子: C, 原子序数: 6, H数: 1

键信息:
  键: 0 -> 1, 类型: SINGLE
  键: 1 -> 2, 类型: DOUBLE
  键: 1 -> 3, 类型: SINGLE
  键: 3 -> 4, 类型: SINGLE
  键: 4 -> 5, 类型: AROMATIC
  键: 5 -> 6, 类型: AROMATIC
  键: 6 -> 7, 类型: AROMATIC
  键: 7 -> 8, 类型: SINGLE
  键: 7 -> 9, 类型: AROMATIC
  键: 9 -> 10, 类型: AROMATIC
  键: 10 -> 4, 类型: AROMATIC

分子表示格式转换:

InChI: InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)
InChIKey: RZVAJINKPMORJF-UHFFFAOYSA-N
分子式: C8H9NO2
分子量: 151.16 Da
LogP (油水分配系数): 1.35
氢键供体数: 2
氢键受体数: 2

============================================================
RDKit基础操作演示完成！
============================================================
```



### 任务1：分子相似性计算 (03_tanimoto_similarity.py)

使用Morgan指纹计算分子间的Tanimoto相似度，比较扑热息痛与多个分子的结构相似性。

```bash
python scripts/03_tanimoto_similarity.py
```
#### 运行结果示例
```bash
============================================================
任务1：分子相似性比较 (Tanimoto相似度)
============================================================

查询分子: 扑热息痛 (Paracetamol)
SMILES: CC(=O)Nc1ccc(O)cc1

------------------------------------------------------------
分子名称               相似度        解读
------------------------------------------------------------
扑热息痛 (自身)       1.0000     非常相似（同系物）
阿司匹林              0.2286     弱相似（有一些共同特征）
布洛芬                0.1892     弱相似（有一些共同特征）
咖啡因                0.1026     很不相似（结构差异大）
葡萄糖                0.0286     很不相似（结构差异大）

------------------------------------------------------------
相似度排序（从高到低）:
  1. 扑热息痛 (自身)       1.0000
  2. 阿司匹林              0.2286
  3. 布洛芬                0.1892
  4. 咖啡因                0.1026
  5. 葡萄糖                0.0286

============================================================
结果分析
============================================================

    1. 扑热息痛 vs 自身 → 相似度 = 1.000（完全一致，验证计算正确性）

    2. 最相似: 阿司匹林 → 0.2286
       两者都含有苯环和羰基，有部分共同的结构特征

    3. 最不相似: 葡萄糖 → 0.0286
       结构类型完全不同，相似度很低

------------------------------------------------------------
💡 应用提示:
   - 虚拟筛选: 通常使用 0.4-0.7 作为相似度阈值
   - 多样性选择: 使用 0.3-0.5 来排除过于相似的分子
   - 相似性搜索: 一般用 0.6 以上作为命中标准
============================================================
```

### 任务2：分子图表示 (04_graph_representation.py)

学习如何将分子"翻译"成GNN能理解的语言：

- ✅ 节点特征：原子类型（C, N, O → 独热编码）
- ✅ 边列表：原子间的连接关系（谁连谁）
- ✅ 边特征：化学键类型（单键/双键/芳香键）
- ✅ 为图神经网络准备标准化的输入格式

```bash
python scripts/04_graph_representation.py
```
#### 运行结果示例
```bash
============================================================
任务2：分子图的边列表表示
============================================================

分子: 扑热息痛 (Paracetamol)
SMILES: CC(=O)Nc1ccc(O)cc1

原子类型: ['C', 'N', 'O']
独热编码维度: 3

------------------------------------------------------------
节点特征 (Node Features)
------------------------------------------------------------
形状: (11, 3)  (原子数=11, 特征维度=3)

每个原子对应的独热编码:
  原子 0: C -> [1.0, 0.0, 0.0]
  原子 1: C -> [1.0, 0.0, 0.0]
  原子 2: O -> [0.0, 0.0, 1.0]
  原子 3: N -> [0.0, 1.0, 0.0]
  原子 4: C -> [1.0, 0.0, 0.0]
  原子 5: C -> [1.0, 0.0, 0.0]
  原子 6: C -> [1.0, 0.0, 0.0]
  原子 7: C -> [1.0, 0.0, 0.0]
  原子 8: O -> [0.0, 0.0, 1.0]
  原子 9: C -> [1.0, 0.0, 0.0]
  原子 10: C -> [1.0, 0.0, 0.0]

------------------------------------------------------------
边列表 (Edge List)
------------------------------------------------------------
形状: (22, 2)  (边数=22, 每个边2个节点)

边列表内容 (源节点 -> 目标节点):
  边索引  源节点  目标节点     键类型
--------------------------------------------------
  0        0          1          单键
  1        1          0          单键
  2        1          2          双键
  3        2          1          双键
  4        1          3          单键
  5        3          1          单键
  6        3          4          单键
  7        4          3          单键
  8        4          5          芳香键
  9        5          4          芳香键
  10       5          6          芳香键
  11       6          5          芳香键
  12       6          7          芳香键
  13       7          6          芳香键
  14       7          8          单键
  15       8          7          单键
  16       7          9          芳香键
  17       9          7          芳香键
  18       9          10         芳香键
  19       10         9          芳香键
  20       10         4          芳香键
  21       4          10         芳香键

------------------------------------------------------------
边特征统计 (键类型分布)
------------------------------------------------------------
  单键: 8 条边
  芳香键: 12 条边
  双键: 2 条边
------------------------------------------------------------
原子邻居信息:
  (每个原子连接到哪些原子)
    原子 0 (C): 邻居 -> [1]
    原子 1 (C): 邻居 -> [0, 2, 3]
    原子 2 (O): 邻居 -> [1]
    原子 3 (N): 邻居 -> [1, 4]
    原子 4 (C): 邻居 -> [3, 5, 10]
    原子 5 (C): 邻居 -> [4, 6]
    原子 6 (C): 邻居 -> [5, 7]
    原子 7 (C): 邻居 -> [6, 8, 9]
    原子 8 (O): 邻居 -> [7]
    原子 9 (C): 邻居 -> [7, 10]
    原子 10 (C): 邻居 -> [9, 4]

============================================================
任务2完成！
============================================================
```

### 任务3：溶解度分布可视化 (05_solubility_histogram.py)

绘制溶解度数据集的直方图，了解目标变量的分布情况。

```bash
python scripts/05_solubility_histogram.py
```

### 任务4：指纹 + 前馈神经网络 (06_fingerprint_nn.py)

构建2层前馈神经网络，输入1024维Morgan指纹，预测分子溶解度。

```bash
python scripts/06_fingerprint_nn.py
```

**模型架构：**

```
输入层 (1024) → 隐藏层 (256, ReLU) → 输出层 (1)
```
#### 运行结果示例
```bash
============================================================
任务4：指纹上的前馈神经网络
============================================================

加载数据...
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子
基线损失 (预测训练集均值): 4.9293

创建模型...

模型结构:
FeedForwardNN(
  (fc1): Linear(in_features=1024, out_features=256, bias=True)
  (activation): ReLU()
  (fc2): Linear(in_features=256, out_features=1, bias=True)
)
模型类型: FeedForwardNN
参数量: 262,657
输入形状: (2, 1024)
输出形状: torch.Size([2, 1])

============================================================
开始训练
============================================================
Epoch 10/100 - Train Loss: 0.4789, Val Loss: 1.8476
Epoch 20/100 - Train Loss: 0.1356, Val Loss: 1.7390
Epoch 30/100 - Train Loss: 0.1046, Val Loss: 1.6986
Epoch 40/100 - Train Loss: 0.0964, Val Loss: 1.7267
Epoch 50/100 - Train Loss: 0.0976, Val Loss: 1.7228
Epoch 60/100 - Train Loss: 0.0857, Val Loss: 1.7378
Epoch 70/100 - Train Loss: 0.0892, Val Loss: 1.6914
Epoch 80/100 - Train Loss: 0.0960, Val Loss: 1.6895
Epoch 90/100 - Train Loss: 0.0902, Val Loss: 1.6837
Epoch 100/100 - Train Loss: 0.0920, Val Loss: 1.6364

============================================================
训练完成！
============================================================
最终训练损失: 0.0920
最终验证损失: 1.6364
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/fingerprint_nn_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -1.253368
460               CCCCCO -0.60 -2.579198
891  C(c1ccccc1)c2ccccc2 -4.08 -2.537115
553                CCCCI -2.96 -2.453290
857           c1ccccc1SC -2.39 -2.039152

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/fingerprint_nn.pt

============================================================
任务4完成！
============================================================
训练损失曲线：

![训练损失曲线](outputs/fingerprint_nn_loss_curve.png)
```
### 任务5：SMILES序列编码 (07_smiles_encoding.py)

将SMILES字符串转换为独热编码序列，支持统一长度填充以用于批处理。

```bash
python scripts/07_smiles_encoding.py
```
#### 运行结果示例
```bash
============================================================
任务5：SMILES序列的独热编码
============================================================
SMILES列表: ['CC(=O)Nc1ccc(O)cc1', 'CC', 'CC=O']
所有符号: ()1=CNOc
符号数量: 8
最大序列长度: 18

------------------------------------------------------------
测试: 乙醛 (Acetaldehyde)
------------------------------------------------------------

SMILES: CC=O
编码结果形状: torch.Size([18, 8])
  行数 (序列长度): 18
  列数 (词汇表大小): 8

非零位置 (行, 列) -> 对应的符号:
  每一行代表一个字符位置，列代表该位置的字符
非零位置 (行, 列):
  位置 (0, 4) -> 符号 'C'
  位置 (1, 4) -> 符号 'C'
  位置 (2, 3) -> 符号 '='
  位置 (3, 6) -> 符号 'O'
```
### 任务6：SMILES + RNN (08_smiles_rnn.py)

使用GRU处理SMILES序列，用最后时刻的隐藏状态预测溶解度。

```bash
python scripts/08_smiles_rnn.py
```

**模型架构：**

```
序列输入 → GRU (2层) → 最后隐藏状态 → 线性层 → 预测
```
#### 运行结果示例
```bash
============================================================
任务6：SMILES上的循环神经网络 (RNN)
============================================================

加载数据...
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子

词汇表大小: 32
最大序列长度: 98

创建模型...

模型结构:
RNNModel(
  (rnn): GRU(32, 64, num_layers=2, batch_first=True)
  (lin): Linear(in_features=64, out_features=1, bias=True)
)
模型类型: RNNModel
参数量: 43,841
输入形状: (2, 98, 32)
输出形状: torch.Size([2, 1])

基线损失 (预测训练集均值): 4.9293

============================================================
开始训练 RNN
============================================================
Epoch 10/100 - Train Loss: 1.5634, Val Loss: 1.9758
Epoch 20/100 - Train Loss: 0.8669, Val Loss: 1.2761
Epoch 30/100 - Train Loss: 0.7759, Val Loss: 0.9584
Epoch 40/100 - Train Loss: 0.6607, Val Loss: 0.9199
Epoch 50/100 - Train Loss: 0.5289, Val Loss: 0.8023
Epoch 60/100 - Train Loss: 0.4768, Val Loss: 0.7827
Epoch 70/100 - Train Loss: 0.4065, Val Loss: 0.7339
Epoch 80/100 - Train Loss: 0.3600, Val Loss: 0.7659
Epoch 90/100 - Train Loss: 0.3481, Val Loss: 0.5958
Epoch 100/100 - Train Loss: 0.3108, Val Loss: 0.6099

============================================================
训练完成！
============================================================
最终训练损失: 0.3108
最终验证损失: 0.6099
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/smiles_rnn_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -0.786607
460               CCCCCO -0.60 -0.689636
891  C(c1ccccc1)c2ccccc2 -4.08 -3.756107
553                CCCCI -2.96 -2.763118
857           c1ccccc1SC -2.39 -1.691364

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/smiles_rnn.pt

============================================================
任务6完成！
============================================================
```
### 任务7：SMILES + CNN (09_smiles_cnn.py)

使用1D卷积提取SMILES序列的局部特征，通过池化聚合后进行预测。

```bash
python scripts/09_smiles_cnn.py
```

**模型架构：**

```
序列输入 → Conv1D → 池化 → 线性层 → 预测
```
#### 运行结果示例-CNNModel
```bash
============================================================
任务7：SMILES上的卷积神经网络 (CNN)
============================================================

加载数据...
正在下载数据: https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv
✅ 数据已保存: data/delaney-processed.csv
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子

词汇表大小: 32
最大序列长度: 98

创建模型...

模型结构:
CNNModel(
  (conv1d): Conv1d(32, 32, kernel_size=(5,), stride=(1,), padding=(2,))
  (pool): AdaptiveAvgPool1d(output_size=1)
  (lin): Linear(in_features=32, out_features=1, bias=True)
)
模型类型: CNNModel
参数量: 5,185
输入形状: (2, 98, 32)
输出形状: torch.Size([2, 1])

基线损失 (预测训练集均值): 4.9293

============================================================
开始训练 CNN
============================================================
Epoch 10/100 - Train Loss: 2.4620, Val Loss: 3.3888
Epoch 20/100 - Train Loss: 1.7528, Val Loss: 2.6004
Epoch 30/100 - Train Loss: 1.4323, Val Loss: 2.1003
Epoch 40/100 - Train Loss: 1.2878, Val Loss: 1.8428
Epoch 50/100 - Train Loss: 1.1847, Val Loss: 1.7013
Epoch 60/100 - Train Loss: 1.1326, Val Loss: 1.6273
Epoch 70/100 - Train Loss: 1.0918, Val Loss: 1.5811
Epoch 80/100 - Train Loss: 1.0738, Val Loss: 1.5508
Epoch 90/100 - Train Loss: 1.0565, Val Loss: 1.5213
Epoch 100/100 - Train Loss: 1.0591, Val Loss: 1.5026

============================================================
训练完成！
============================================================
最终训练损失: 1.0591
最终验证损失: 1.5026
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/smiles_cnn_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -1.418692
460               CCCCCO -0.60 -1.189165
891  C(c1ccccc1)c2ccccc2 -4.08 -4.551840
553                CCCCI -2.96 -2.281808
857           c1ccccc1SC -2.39 -2.754482

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/smiles_cnn.pt

============================================================
任务7完成！
============================================================
```
#### 运行结果示例-CNNModelDeep
```bash
============================================================
任务7：SMILES上的卷积神经网络 (CNN)
============================================================

加载数据...
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子

词汇表大小: 32
最大序列长度: 98

创建模型...

模型结构:
CNNModelDeep(
  (convs): Sequential(
    (0): Conv1d(32, 32, kernel_size=(5,), stride=(1,), padding=(2,))
    (1): ReLU()
    (2): Conv1d(32, 32, kernel_size=(5,), stride=(1,), padding=(2,))
    (3): ReLU()
    (4): Conv1d(32, 32, kernel_size=(5,), stride=(1,), padding=(2,))
  )
  (pool): AdaptiveAvgPool1d(output_size=1)
  (lin): Linear(in_features=32, out_features=1, bias=True)
)
模型类型: CNNModelDeep
参数量: 15,489
输入形状: (2, 98, 32)
输出形状: torch.Size([2, 1])

基线损失 (预测训练集均值): 4.9293

============================================================
开始训练 CNN
============================================================
Epoch 10/100 - Train Loss: 1.0166, Val Loss: 1.4446
Epoch 20/100 - Train Loss: 0.7329, Val Loss: 1.0825
Epoch 30/100 - Train Loss: 0.6374, Val Loss: 0.9903
Epoch 40/100 - Train Loss: 0.5664, Val Loss: 0.8959
Epoch 50/100 - Train Loss: 0.5098, Val Loss: 0.8216
Epoch 60/100 - Train Loss: 0.4846, Val Loss: 0.7571
Epoch 70/100 - Train Loss: 0.4443, Val Loss: 0.7224
Epoch 80/100 - Train Loss: 0.4285, Val Loss: 0.7766
Epoch 90/100 - Train Loss: 0.3965, Val Loss: 0.6878
Epoch 100/100 - Train Loss: 0.3504, Val Loss: 0.7318

============================================================
训练完成！
============================================================
最终训练损失: 0.3504
最终验证损失: 0.7318
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/smiles_cnn_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -1.259114
460               CCCCCO -0.60 -1.311722
891  C(c1ccccc1)c2ccccc2 -4.08 -4.030847
553                CCCCI -2.96 -3.341196
857           c1ccccc1SC -2.39 -2.156257

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/smiles_cnn.pt

============================================================
任务7完成！
============================================================
```
### CNN 模型对比

| 指标 | 单层 CNN | 3层 CNN | RNN |
|------|----------|---------|-----|
| 验证损失 | 1.50 | **0.73** | 0.61 |
| 参数量 | 5,185 | 15,489 | 43,841 |
| 感受野 | 5 个字符 | 13 个字符 | — |

### 训练收敛趋势

| Epoch | 单层 CNN | 3层 CNN |
|-------|----------|---------|
| 10 | 3.39 | 1.44 |
| 50 | 1.70 | 0.82 |
| 100 | 1.50 | 0.73 |

### 预测结果对比

| SMILES | 真实值 | 单层 CNN | 3层 CNN |
|--------|--------|----------|---------|
| CC(C)(C)CCO | -0.50 | -1.42 | -1.26 |
| CCCCCO | -0.60 | -1.19 | -1.31 |
| C(c1ccccc1)c2ccccc2 | -4.08 | -4.55 | **-4.03** |
| CCCCI | -2.96 | -2.28 | **-3.34** |
| c1ccccc1SC | -2.39 | -2.75 | -2.16 |

### 关键结论

- 3层 CNN 验证损失比单层 CNN **降低 51%**（1.50 → 0.73）
- 参数量仅为 RNN 的 **1/3**，效果接近 RNN（0.73 vs 0.61）
- 感受野从 5 扩大到 13 个字符，能完整覆盖苯环等复杂结构
- 在含苯环的复杂分子预测上，3层 CNN 明显优于单层 CNN

### 任务8：图数据结构 (10_graphs_class.py)

演示Graphs类的核心功能：从SMILES创建图、合并多个图以支持批处理。

```bash
python scripts/10_graphs_class.py
```
#### 运行结果示例
```bash
============================================================
任务8：图数据结构 (Graphs类)
============================================================
丙酮 - 节点数: 4, 边数: 6
乙烷 - 节点数: 2, 边数: 2

合并后:
  总节点数: 6
  总边数: 8
  图数量: 2
  节点到图的映射: [0, 0, 0, 0, 1, 1]

节点特征形状: torch.Size([6, 72])
边列表形状: torch.Size([8, 2])
边特征形状: torch.Size([8, 4])
```

### 任务9：图神经网络 (11_graph_gnn.py)

基于门控图神经网络（Gated Graph Neural Network），使用GRU进行消息传递。

```bash
python scripts/11_graph_gnn.py
```

**模型架构：**

```
节点特征 → 消息传递 (4步) → 注意力聚合 → 线性层 → 预测
```
#### 运行结果示例
```bash
============================================================
任务9：图神经网络 (GNN)
============================================================

加载数据...
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子

创建模型...

节点特征维度: 72
消息传递步数: 4

模型结构:
GNN(
  (message_projection): Linear(in_features=72, out_features=72, bias=False)
  (update_gru): GRUCell(72, 72)
  (attn_net): Linear(in_features=72, out_features=1, bias=True)
  (proj_net): Linear(in_features=72, out_features=72, bias=True)
  (final_lin): Linear(in_features=72, out_features=1, bias=True)
)
模型类型: GNN
参数量: 42,122

基线损失 (预测训练集均值): 4.9293

============================================================
开始训练 GNN
============================================================
Epoch 10/100 - Train Loss: 1.1885, Val Loss: 1.6876
Epoch 20/100 - Train Loss: 0.9663, Val Loss: 1.5642
Epoch 30/100 - Train Loss: 0.9368, Val Loss: 1.4211
Epoch 40/100 - Train Loss: 0.8252, Val Loss: 1.2304
Epoch 50/100 - Train Loss: 0.7565, Val Loss: 1.1189
Epoch 60/100 - Train Loss: 0.6816, Val Loss: 1.0316
Epoch 70/100 - Train Loss: 0.6622, Val Loss: 1.0473
Epoch 80/100 - Train Loss: 0.6515, Val Loss: 0.9930
Epoch 90/100 - Train Loss: 0.6302, Val Loss: 0.9918
Epoch 100/100 - Train Loss: 0.6421, Val Loss: 0.8244

============================================================
训练完成！
============================================================
最终训练损失: 0.6421
最终验证损失: 0.8244
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/graph_gnn_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -1.316153
460               CCCCCO -0.60 -0.754125
891  C(c1ccccc1)c2ccccc2 -4.08 -5.475595
553                CCCCI -2.96 -2.490324
857           c1ccccc1SC -2.39 -2.255278

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/graph_gnn.pt

============================================================
任务9完成！
============================================================
```
### 任务9扩展：增强版图神经网络 (12_graph_gnn_enhanced.py)

在基础版 GNN 上做了两项关键改进：

-  **边特征**：区分单键/双键/三键/芳香键、是否在环中、是否共轭（8维）
-  **节点特征**：在原72维原子类型基础上，增加度数、氢数、电荷、环、芳香性（92维）
-  **动态变换矩阵**：每条边根据自身特征生成变换矩阵，融合边信息和节点消息

#### 运行方式
```bash
python scripts/12_graph_gnn_enhanced.py
```
#### 模型架构
```bash
节点特征(92) → 消息传递(4步, 带边特征) → 注意力聚合 → 线性层 → 预测
```
#### 运行结果示例
```bash
============================================================
任务9扩展：增强版图神经网络 (GNN Enhanced)
============================================================

加载数据...
Failed to patch pandas - unable to change molecule rendering
训练集: 1016 个分子
验证集: 112 个分子

创建模型...
节点特征维度: 92

模型结构:
GNNEnhanced(
  (message_projection): Linear(in_features=92, out_features=92, bias=False)
  (edge_projection): Linear(in_features=8, out_features=8464, bias=False)
  (update_gru): GRUCell(92, 92)
  (attn_net): Linear(in_features=92, out_features=1, bias=True)
  (proj_net): Linear(in_features=92, out_features=92, bias=True)
  (final_lin): Linear(in_features=92, out_features=1, bias=True)
)
模型类型: GNNEnhanced
参数量: 136,254

基线损失: 4.9293

============================================================
开始训练 GNN Enhanced
============================================================
  Epoch 1: 新最佳验证损失 2.3225
  Epoch 2: 新最佳验证损失 2.1679
  Epoch 3: 新最佳验证损失 1.7360
  Epoch 4: 新最佳验证损失 1.6099
  Epoch 5: 新最佳验证损失 1.3845
  Epoch 6: 新最佳验证损失 1.2458
  Epoch 8: 新最佳验证损失 1.1156
  Epoch 9: 新最佳验证损失 0.8889
Epoch 10/100 - Train Loss: 0.6446, Val Loss: 0.9417
  Epoch 14: 新最佳验证损失 0.8394
  Epoch 15: 新最佳验证损失 0.8053
  Epoch 16: 新最佳验证损失 0.6809
Epoch 20/100 - Train Loss: 0.5603, Val Loss: 1.0468
  Epoch 21: 新最佳验证损失 0.5948
  Epoch 27: 新最佳验证损失 0.5925
  Epoch 29: 新最佳验证损失 0.5692
Epoch 30/100 - Train Loss: 0.3080, Val Loss: 0.6687
  Epoch 35: 新最佳验证损失 0.5422
Epoch 40/100 - Train Loss: 0.2427, Val Loss: 0.7277
  Epoch 42: 新最佳验证损失 0.5410
Epoch 50/100 - Train Loss: 0.1765, Val Loss: 0.6813

🛑 早停触发！已经 10 轮无改善
   最佳验证损失: 0.5410

✅ 已加载最佳模型（验证损失 0.5410）

============================================================
训练完成！
============================================================
最终训练损失: 0.1655
最终验证损失: 0.6998
基线损失: 4.9293
✅ 模型表现优于基线！
✅ 损失曲线已保存: /root/autodl-tmp/ML4Chem/outputs/graph_gnn_enhanced_loss_curve.png

验证集预测结果（前5个）:
                  smiles   真实值       预测值
690          CC(C)(C)CCO -0.50 -1.569031
460               CCCCCO -0.60 -1.316876
891  C(c1ccccc1)c2ccccc2 -4.08 -3.784334
553                CCCCI -2.96 -2.676327
857           c1ccccc1SC -2.39 -2.464561

✅ 模型权重已保存到: /root/autodl-tmp/ML4Chem/models/graph_gnn_enhanced.pt

============================================================
任务9扩展完成！
============================================================
```
#### 关键结论
- 增强版 GNN 验证损失 0.54，超越 RNN (0.61)，成为所有模型中最佳
- 相比基础版 GNN (0.82) 提升 34%
- 代价：参数量从 42K 增加到 136K（约 3 倍）
- 采用早停策略，防止过拟合（未早停时最终损失 0.77）


#### 模型效果对比
| 模型 | 验证损失 | 参数量 |
|------|----------|--------|
| 🏆 **GNN Enhanced** | **0.54** | 136K |
| 🥈 RNN | 0.61 | 44K |
| 🥉 CNN 3层 | 0.73 | 15K |
| GNN 基础版 | 0.82 | 42K |
| CNN 单层 | 1.50 | 5K |
| 指纹NN | 1.55 | 263K |
| 基线 | 4.93 | 0 |



## 🔧 核心模块说明

### featurizers.py - 特征化器

| 类/函数 | 功能 | 用途 |
|---------|------|------|
| `SymbolFeaturizer` | 符号→独热编码 | 原子类型、键类型、SMILES字符编码 |
| `morgan_fp_from_smiles` | SMILES→Morgan指纹 | 任务4的特征提取 |
| `tanimoto_similarity` | 计算分子相似度 | 任务1 |
| `mol_to_edge_list_graph` | 分子→图表示 | 任务2、任务8、任务9 |

### models.py - 模型定义

| 模型 | 输入 | 输出 | 对应任务 |
|------|------|------|----------|
| `FeedForwardNN` | 指纹向量 (1024维) | 溶解度标量 | 任务4 |
| `RNNModel` | 独热编码序列 | 溶解度标量 | 任务6 |
| `CNNModel` | 独热编码序列 | 溶解度标量 | 任务7 |
| `GNN` | 分子图 | 溶解度标量 | 任务9 |

### graphs.py - 图数据结构

核心功能：

- `from_smiles()`: 从SMILES创建单个分子图
- `concatenate()`: 合并多个图以支持批处理
- `num_graphs`: 获取图中包含的分子数量

### data_utils.py - 数据工具

- `load_solubility_data()`: 加载溶解度数据集
- `split_train_val()`: 划分训练集和验证集

### train_utils.py - 训练工具

- `plot_train_val_loss()`: 绘制训练曲线
- `compute_dummy_baseline()`: 计算基线MSE
- `print_model_summary()`: 打印模型摘要

## 🧪 测试

```bash
# 安装pytest
pip install pytest

# 运行测试
pytest tests/
```

## 📚 延伸阅读

- [RDKit官方文档](https://www.rdkit.org/docs/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [Graph Neural Networks综述](https://arxiv.org/abs/1812.08434)

**Happy Learning! 🎓**
