# Step 25：BERT Intent Classifier 训练与可选接入

## 目标

本步骤把 Query Understanding 从纯规则 baseline 扩展为可选 Hybrid 模式：

```text
规则抽 slots
+ BERT 判断 intent
+ LLM 仍作为复杂表达的可选 fallback
```

注意：BERT 目前只负责意图分类，不负责预算、品牌、品类、商品指代等结构化 slots。slots 继续由 Step 24 的 `intent_rules.json` 规则系统抽取。

## GPU 环境

当前机器检测到：

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
Driver CUDA: 13.1
```

可用训练环境：

```text
D:\anaconda\envs\py3.10\python.exe
torch 2.11.0+cu128
cuda: true
transformers: 5.9.0
sklearn: 1.7.2
```

`D:\anaconda\python.exe` 是 CPU 版 PyTorch，不适合训练。

## 新增文件

```text
scripts/train_query_intent_bert.py
app/services/intent_classifier.py
requirements-ml.txt
reports/query_understanding_bert_metrics_latest.json
reports/query_understanding_bert_predictions_latest.jsonl
steps/step-25-bert-intent-classifier.md
```

模型权重输出到：

```text
training/query_intent_bert/
```

该目录已加入 `.gitignore`，不会把几百 MB 的模型权重提交到 GitHub。

## 训练命令

```powershell
D:\anaconda\envs\py3.10\python.exe scripts\train_query_intent_bert.py `
  --model-name bert-base-chinese `
  --epochs 12 `
  --batch-size 16 `
  --learning-rate 5e-5 `
  --device cuda
```

第一轮试过小模型：

```text
uer/chinese_roberta_L-2_H-128
```

但 5 个 epoch 没有收敛，基本只预测多数类。所以最终采用：

```text
bert-base-chinese
```

## 训练结果

在当前 800 条 train、200 条 eval 合成增强数据上：

```text
accuracy: 1.0
macro_f1: 1.0
avg_latency_ms: 0.6347
elapsed_seconds: 125.52
device: cuda
```

这说明 BERT 对当前合成模板数据已经能完全拟合，但也必须说明风险：

```text
这个分数不能直接代表真实线上泛化能力。
后续需要人工标注 eval set 或真实日志验证。
```

## 模型对比

重新运行：

```powershell
python scripts\evaluate_query_understanding_models.py `
  --bert-predictions reports\query_understanding_bert_predictions_latest.jsonl
```

当前对比：

```text
rule_baseline             intent_macro_f1 = 0.7147, slot_f1 = 0.5963
char_ngram_nb_classifier  intent_macro_f1 = 0.9861, slot_f1 = 0
distilbert_classifier     intent_macro_f1 = 1.0,    slot_f1 = 0
```

推荐结论：

```text
BERT/DistilBERT 负责高频 intent 分类
rule_baseline 负责 slots 抽取和兜底
```

## 在线接入

新增：

```text
app/services/intent_classifier.py
```

默认不开启，不影响当前 demo 稳定性。

开启方式：

```powershell
$env:CHAT_INTENT_MODEL_ENABLED="true"
$env:CHAT_INTENT_MODEL_PATH="D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step\training\query_intent_bert"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

开启后链路变为：

```text
用户输入
-> 规则抽 slots
-> BERT 覆盖 intent
-> source = bert+rule_slots
-> ToolRouter
-> LangGraph 推荐链路
```

如果模型不存在、加载失败或置信度低于阈值：

```text
自动回退到 rule intent
```

## 面试表达

可以这样讲：

> 我没有直接用 BERT 替代整个 Query Understanding，而是把任务拆开：intent classification 用 BERT，因为它高频、低延迟、成本稳定；slots extraction 继续用规则，因为预算、品牌、品类、指代这类结构化字段需要可控和可解释；复杂表达或低置信度再交给 LLM。这样既能展示模型训练能力，也保留了线上系统的稳定 fallback。
