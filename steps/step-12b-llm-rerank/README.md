# Step 12b: LLM 商品重排 Agent

## 这一阶段解决什么问题

Step 12a 之前，商品排序还是纯规则：

```text
类目 + 品牌 + 标签 + 预算 + 评分 - 最近浏览 - 点踩
```

Step 12b 在规则排序前增加 LLM 重排路径：

```text
Chroma 粗召回候选商品
  -> ProductRecAgent rerank
  -> 读取 effective_request.context["llm_hint"]
  -> LLM 输出商品 ID 排序
  -> 失败时回退 score_product 规则排序
```

## 触发条件

只有当请求里存在：

```text
request.context["llm_hint"]
```

才会尝试 LLM 重排。这个字段来自 Step 11 的 `UserProfileAgent`。

## LLM 输入

```text
用户偏好类目
用户偏好品牌
用户偏好标签
预算范围
recommendation_hint
候选商品 ID / 名称 / 类目 / 价格 / 品牌 / 标签
```

## LLM 输出

LLM 只需要输出商品 ID 数组：

```json
["B07ZPSG8P5", "B018WY6OSW", "B075VNFGBH"]
```

项目会做：

```text
1. 校验 product_id 是否真实存在
2. 去重
3. 如果数量不足，用原候选顺序补齐
4. 转换成 AgentResult.data.product_ids
```

## 降级逻辑

如果出现这些情况：

```text
LLM 不可用
LLM 超时
LLM 返回不是 JSON 数组
LLM 返回的 product_id 都无效
```

就自动回到原来的：

```text
score_product() 规则排序
```

## 接口返回里怎么看

推荐接口：

```text
POST /api/v1/recommend
```

重点看：

```text
agent_results.product_rerank.data.mode
agent_results.product_rerank.data.backend
agent_results.product_rerank.data.scores
```

LLM 成功：

```json
{
  "mode": "llm_rerank",
  "backend": "llm+rule_rerank"
}
```

规则 fallback：

```json
{
  "mode": "rerank",
  "backend": "rule_rerank"
}
```

## 前端变化

LLM 用户画像面板新增：

```text
重排模式
文案模式
```

这样可以直接看到当前请求是否走了 `llm_rerank`。

## 验证方式

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
D:\anaconda\envs\py3.10\python.exe -m compileall app tests
```

已验证：

```text
17 passed
compileall 通过
```

## 推荐阅读顺序

```text
app/agents/product_rec_agent.py
app/personalization.py
app/static/index.html
tests/test_recommender.py
```
