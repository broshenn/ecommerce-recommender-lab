# Step 8：A/B 测试 + Metrics 指标统计

## 这一步要解决什么

Step 7 已经把推荐流程拆成了 Supervisor + 4 个 Agent，但它还只能“给出推荐结果”。真实业务系统还需要知道：

```text
这个用户进了哪个实验组？
这次推荐调用了哪些 Agent？
每个 Agent 是否成功？
每个 Agent 花了多久？
推荐接口一共被调用了多少次？
```

所以 Step 8 加了两个工程能力：

```text
ABTestEngine      -> 根据 user_id 做稳定分桶
MetricsCollector -> 记录 Agent 调用指标和业务事件
```

## 新增文件

```text
app/services/__init__.py
app/services/ab_test.py
app/services/metrics.py
```

## 改动文件

```text
app/models.py
app/main.py
app/orchestrator/supervisor.py
app/static/index.html
tests/test_recommender.py
README.md
steps/README.md
```

## A/B 测试怎么做

当前实验 ID：

```text
recommendation_strategy_v1
```

当前有两个组：

```text
control   -> 当前规则排序策略
treatment -> 预留增强推荐策略入口
```

现在两个组还没有真正执行不同推荐算法。这样设计是为了先把“实验分桶能力”接好，后面接 Chroma、LLM 重排、训练排序模型时，就可以比较不同策略效果。

分桶方式：

```text
sha256(experiment_id:user_id) -> bucket -> control / treatment
```

同一个 `user_id` 每次都会进入同一个实验组。

## Metrics 统计了什么

当前记录：

```text
Agent 调用次数
Agent 成功次数
Agent 失败次数
Agent 成功率
Agent 平均耗时
Agent 最近一次耗时
业务事件次数
```

当前业务事件：

```text
recommend_request
recommend_success
```

## 新增接口

查看实验配置：

```http
GET /api/v1/experiments
```

查看某个用户的实验分桶：

```http
GET /api/v1/experiments?user_id=u001
```

查看指标：

```http
GET /api/v1/metrics
```

## 推荐响应新增字段

`POST /api/v1/recommend` 现在会多返回：

```json
{
  "experiment_group": "control",
  "experiment": {
    "experiment_id": "recommendation_strategy_v1",
    "group": "control",
    "reason": "user_id 稳定哈希分桶，bucket=12.34",
    "config": {
      "strategy": "rule_ranking"
    }
  }
}
```

## 为什么这一步重要

这一步让项目从“能推荐”往“能观察、能实验”走了一步。后面每增加一个推荐策略，都可以挂到实验组里，再通过 Metrics 和后续点击/加购/转化指标判断是否真的更好。

## 运行

```powershell
cd D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

## 测试

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

当前验证：

```text
14 passed
```

## 下一步

Step 9 建议接入 Chroma，把 `ProductRecAgent` 的召回阶段升级为“商品文本向量召回”，再保留当前规则打分做重排。
