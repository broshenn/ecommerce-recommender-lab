# Step 23：Query Understanding 模型对比

## 目标

本步骤用于回答面试中经常会被追问的问题：

- 为什么不直接全部调用大模型？
- BERT/DistilBERT 在这里到底起什么作用？
- 没有真实线上数据时训练数据从哪里来？
- 规则、LLM、BERT 该怎么取舍？

结论不是“为了用模型而用模型”，而是把 Query Understanding 拆成两个子问题：

```text
意图分类：判断用户要推荐、补槽、比较、解释、反馈、问商品还是闲聊。
槽位抽取：抽预算、类目、品牌、标签、反馈事件和商品指代。
```

## 数据来源

新增并纳入 Git 的数据：

- `data/query_understanding_train.jsonl`：800 条训练样本
- `data/query_understanding_eval.jsonl`：200 条评测样本

数据来自两部分：

- DeepSeek 合成种子样本
- 基于商品目录的模板增强

数据字段：

```json
{
  "text": "我想买200元以内的键盘",
  "intent": "recommend_products",
  "category": "办公",
  "slots": {
    "budget_min": null,
    "budget_max": 200,
    "preferred_categories": ["办公"],
    "liked_brands": [],
    "preferred_tags": ["键盘"],
    "event_type": "",
    "product_refs": []
  },
  "need_recommendation": true
}
```

## 脚本

### 数据生成

- `scripts/generate_query_understanding_synthetic.py`
  - 调 DeepSeek/OpenAI-compatible LLM 生成新样本。
  - 需要 LLM API key。

- `scripts/augment_query_understanding_synthetic.py`
  - 读取少量种子样本和商品目录。
  - 通过模板扩增到训练/评测集。
  - 不依赖外部 API。

### 模型对比

新增：

```text
scripts/evaluate_query_understanding_models.py
```

默认运行：

```powershell
python scripts\evaluate_query_understanding_models.py
```

输出：

```text
reports/query_understanding_model_compare_latest.json
reports/query_understanding_model_compare_latest.md
```

## 当前对比模型

| 模型 | 状态 | 说明 |
|------|------|------|
| `rule_baseline` | 默认必跑 | 当前 `IntentAgent` 规则兜底，能做意图和槽位 |
| `char_ngram_nb_classifier` | 默认必跑 | 轻量可训练意图分类器，纯 Python，无额外依赖 |
| `llm_classifier` | 可选 | 加 `--include-llm` 后调用配置的大模型 |
| `distilbert_classifier` | 可选 | 通过 `--bert-predictions` 评估外部 BERT/DistilBERT 预测结果 |

## 当前结果

200 条 eval 上：

| 模型 | Intent Acc | Intent Macro F1 | Slot F1 | Avg Latency |
|------|------------|-----------------|---------|-------------|
| `rule_baseline` | 0.645 | 0.7147 | 0.5963 | 9.3174 ms |
| `char_ngram_nb_classifier` | 0.985 | 0.9861 | 0.0 | 0.0582 ms |

解释：

- `char_ngram_nb_classifier` 意图分类很强，因为它专门学习 intent。
- 它的 `slot_f1 = 0`，因为它不做槽位抽取。
- `rule_baseline` 意图分类较弱，但可以抽预算、品类、品牌和标签。

## 业务结论

当前推荐：

```text
高频入口：训练型意图分类器 / 后续 DistilBERT 做 intent
槽位抽取：继续用规则，或者后续训练序列标注模型
复杂表达：可选 LLM classifier
线上兜底：规则 baseline 必须保留
```

换句话说，BERT/DistilBERT 在这里最适合做：

```text
用户意图分类
```

不建议第一版让 BERT 单独负责完整 Query Understanding，因为预算、品牌、标签、商品指代这些结构化字段还需要规则、NER 或序列标注模型配合。

## 为什么不直接用大模型

| 维度 | LLM | BERT/DistilBERT | 规则 |
|------|-----|-----------------|------|
| 复杂语言理解 | 强 | 中强 | 弱 |
| 延迟 | 高 | 低 | 极低 |
| 成本 | 高 | 低 | 极低 |
| 可控性 | 中 | 高 | 高 |
| 槽位精确抽取 | 强但需校验 | 需要额外模型 | 中 |
| 线上兜底 | 不适合唯一兜底 | 适合高频分类 | 必须保留 |

## 面试表达

可以这样讲：

> Query Understanding 我没有直接绑定某一种模型，而是做了对比实验。规则 baseline 稳定、可控、能抽槽位，但泛化能力有限；训练型分类器在合成数据上意图 F1 很高，延迟也低，适合高频入口；LLM 更适合理解复杂表达，但成本和延迟不适合每次都调用；BERT/DistilBERT 可以作为后续高频 intent classifier，但槽位抽取仍需要规则或序列标注模型。所以当前工程上保留规则 fallback，模型实验作为增强路径。

## 下一步

如果继续增强：

- 使用 `query_understanding_train.jsonl` 微调 DistilBERT intent classifier。
- 增加 slot sequence labeling 数据，例如 BIO 标注预算、品牌、类目、商品指代。
- 把 LLM 作为 teacher，持续蒸馏复杂表达到小模型。
- 引入人工校验集，避免只在合成数据上过拟合。
