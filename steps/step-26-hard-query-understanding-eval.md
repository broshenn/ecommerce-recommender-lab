# Step 26：人工 Hard Eval 与 BERT 泛化验证

## 目标

Step 25 的 BERT 在合成 eval set 上达到：

```text
accuracy = 1.0
macro_f1 = 1.0
```

这个结果说明模型能拟合当前合成模板，但不能直接证明真实泛化能力。因此本步骤新增一份“人工模拟”的 hard eval，用更口语、更绕、更接近真实用户的表达来验证 Query Understanding。

## 新增文件

```text
data/query_understanding_hard_eval.jsonl
scripts/evaluate_query_understanding_hard.py
reports/query_understanding_hard_eval_latest.json
reports/query_understanding_hard_eval_latest.md
steps/step-26-hard-query-understanding-eval.md
```

## Hard Eval 覆盖场景

共 36 条人工模拟样本，覆盖：

```text
single_recommend        单轮推荐
multi_slot              多轮补槽
goal_switch             目标切换
negative_feedback       负反馈
compare                 商品比较
explain                 解释推荐
ask_product             商品信息问答
smalltalk               闲聊/元问题
unsupported_catalog     商品库边界
edge_mixed              混合干扰表达
```

例子：

```text
我预算就两百，通勤路上用，别太夹耳
第二个太贵了，换个便宜点的
刚才第二个和那个 Sony 的差在哪
先别推荐，我只是测试一下对话框
电脑不是配件，我要整机，不过库里没有就直接告诉我
```

## 评测命令

```powershell
python scripts\evaluate_query_understanding_hard.py
```

## 当前结果

| 模型 | Hard Intent Acc | Hard Intent Macro F1 | Slot F1 | Smalltalk Guard | Feedback Acc | Ref Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rule_baseline` | 0.6944 | 0.6920 | 0.7634 | 1.0 | 0.5 | 1.0 |
| `char_ngram_nb_classifier` | 0.7222 | 0.7315 | 0.0 | 0.75 | 1.0 | 0.0 |
| `bert_rule_slots` | 0.8889 | 0.9026 | 0.7634 | 1.0 | 1.0 | 1.0 |

## 关键发现

### 1. 合成集满分不等于真实泛化

BERT 在合成 eval 上是满分，但 hard eval 后变成：

```text
hard_intent_macro_f1 = 0.9026
```

这更可信，也更适合面试说明：

```text
我没有只拿合成数据自证，而是专门构造 hard set 检查泛化。
```

### 2. BERT 适合 intent，不适合单独做完整理解

BERT 的 intent 明显强于规则：

```text
rule_baseline hard_intent_macro_f1 = 0.6920
bert_rule_slots hard_intent_macro_f1 = 0.9026
```

但 slots 仍然来自规则：

```text
bert_rule_slots slot_f1 = rule_baseline slot_f1 = 0.7634
```

这说明当前最佳工程方案仍然是：

```text
BERT 负责 intent
规则负责 slots
```

### 3. Smalltalk guard 必须保留规则保护

初版 hard eval 暴露了一个问题：

```text
BERT 会把部分“不是问商品 / 先别推荐 / 屏幕 / 星期几”误判成业务意图
```

因此新增了线上 guard：

```text
如果规则高置信识别 smalltalk，则 BERT 不覆盖
```

这让 `bert_rule_slots` 的：

```text
smalltalk_guard_rate = 1.0
```

## 当前结论

推荐线上策略：

```text
1. 规则先做 slots 和 smalltalk guard
2. BERT 覆盖高频 intent
3. 低置信度或复杂表达再交给 LLM
4. 所有结果继续走 ToolRouter 和 LangGraph 推荐链路
```

## 仍然暴露的问题

hard eval 还显示一些后续可优化点：

```text
品牌优先 Sony，价格别太离谱
别给我手机配件了，我要办公用的东西
算了不看保护壳了，给我看看耳机
电脑不是配件，我要整机，不过库里没有就直接告诉我
```

这些主要是：

```text
补槽 vs 新推荐的边界
否定表达里的目标切换
商品库不支持场景的显式澄清
```

后续可以继续增强规则、加入更多 hard 样本，或者让 LLM 只处理低置信/冲突样本。

## 面试表达

可以这样讲：

> BERT 在合成数据上分数很高，但我没有直接把它当结论。我又做了一份人工 hard eval，专门覆盖真实口语、闲聊误触发、负反馈、目标切换和商品库边界。结果显示 BERT+规则 slots 的 hard intent macro F1 达到 0.9026，明显强于纯规则，但 slots 仍然要靠规则保证可解释和稳定。因此我的线上设计是 hybrid，而不是单模型替代全部 Query Understanding。
