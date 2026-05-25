# Step 13b: A/B 实验数据闭环

这一阶段把 A/B 实验从“能分组、能切策略”，继续推进到“能记录结果、能看效果”。

## 这一步解决什么问题

Step 13a 之后，系统已经能做到：

```text
control   -> 规则链路
treatment -> LLM 增强链路
```

但还缺一个实验系统最关键的闭环：

```text
用户看到了推荐吗？     -> 曝光 exposure
用户点了推荐商品吗？   -> 点击 click
哪个组 CTR 更高？      -> stats
以后要不要多给好组流量？ -> Thompson Sampling
```

这一步先做内存版统计，方便学习和调试。后面如果要做生产化，再把 `_events` 换成 SQLite、Redis Stream 或 Kafka 都可以。

## 新增能力

### 1. 推荐成功后自动记录曝光

位置：

```text
app/orchestrator/supervisor.py
```

当 `/api/v1/recommend` 成功返回前，Supervisor 会调用：

```text
ab_engine.record_exposure(experiment_id, group, user_id)
```

含义是：这个用户已经看到了某个实验组产出的推荐结果。

### 2. 用户行为回传实验结果

位置：

```text
app/main.py
```

新增接口：

```text
POST /api/v1/experiments/{experiment_id}/outcome
```

请求体：

```json
{
  "experiment_id": "recommendation_strategy_v1",
  "group": "control",
  "user_id": "u001",
  "success": true,
  "product_id": "B07ZPSG8P5"
}
```

这里的 `success=true` 表示一次点击或正向行为；`success=false` 表示跳过、负反馈或不喜欢。

### 3. 实验统计

位置：

```text
app/services/ab_test.py
```

`get_stats()` 会返回每个实验组的：

```text
exposures     曝光次数
clicks        点击次数
skips         负反馈/跳过次数
ctr           clicks / exposures
alpha         Thompson Sampling 成功计数
beta          Thompson Sampling 失败计数
expected_ctr  Beta 分布期望值
```

### 4. Thompson Sampling 计数器

现在每个 variant 都有：

```text
alpha = 1
beta  = 1
```

点击成功时：

```text
alpha += 1
```

负反馈时：

```text
beta += 1
```

`assign_thompson()` 会从每个组的 Beta 分布采样，哪个组采样值高，就把用户分给哪个组。

当前推荐主链路仍然使用稳定 hash 分桶 `assign()`，因为学习阶段更容易复现；`assign_thompson()` 已经作为下一步动态流量分配的入口准备好。

## 前端闭环

位置：

```text
app/static/index.html
```

商品卡片按钮原本只记录用户行为：

```text
查看 / 喜欢 / 不喜欢 / 加购
```

现在会额外回传实验 outcome：

```text
查看、喜欢、加购 -> success=true
不喜欢          -> success=false
```

然后重新请求推荐，形成：

```text
推荐曝光 -> 用户点击/反馈 -> 写入行为 -> 写入实验 outcome -> 重新推荐
```

## 怎么验证

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

当前结果：

```text
22 passed
```

也可以手动验证：

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"ab-user-1\",\"num_items\":2}"

curl -X POST http://127.0.0.1:8010/api/v1/experiments/recommendation_strategy_v1/outcome ^
  -H "Content-Type: application/json" ^
  -d "{\"experiment_id\":\"recommendation_strategy_v1\",\"group\":\"control\",\"user_id\":\"ab-user-1\",\"success\":true,\"product_id\":\"B07ZPSG8P5\"}"

curl http://127.0.0.1:8010/api/v1/experiments
```

## 阅读重点

这一步的核心不是复杂算法，而是推荐系统的实验闭环：

```text
曝光是分母，点击是分子，CTR 是观察指标，Beta 参数是动态调流量的基础。
```

没有曝光记录，点击数没有意义；没有 outcome 回传，A/B 实验只能分组，不能判断效果。
