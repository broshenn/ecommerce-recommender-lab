# Step 9：Chroma 向量召回

## 这一步要解决什么

Step 8 之前，`ProductRecAgent` 的做法是：

```text
拿到全部商品 -> 对每个商品做规则打分 -> 排序
```

这能跑通，但不像真实推荐系统。真实系统通常会拆成两步：

```text
召回 recall   -> 先从海量商品里找一批候选
重排 rerank   -> 再对候选商品精细打分排序
```

所以 Step 9 加入了 Chroma 向量库，把商品召回升级为：

```text
用户画像文本 -> embedding -> Chroma 检索候选商品 -> 规则重排 -> 库存过滤
```

## 新增文件

```text
app/services/vector_store.py
.env.example
steps/step-09-chroma-vector-recall/README.md
```

## 改动文件

```text
app/agents/product_rec_agent.py
app/orchestrator/supervisor.py
app/main.py
requirements.txt
README.md
tests/test_recommender.py
```

## ProductRecAgent 现在有两个模式

召回模式：

```text
mode="recall"
```

优先使用 Chroma 向量召回。

重排模式：

```text
mode="rerank"
```

继续复用当前规则打分：

```text
类目匹配
品牌匹配
标签匹配
预算匹配
评分
最近浏览降权
不喜欢商品强降权
```

## 千问 API 配置

项目根目录已经创建：

```text
.env
.env.example
```

模板内容：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=local
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
```

要调用千问，把它改成：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=qwen
DASHSCOPE_API_KEY=你的真实DashScopeKey
```

项目会调用千问 / DashScope 的 OpenAI 兼容 embedding 接口：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
```

当前默认模型：

```text
text-embedding-v4
dimensions=1024
```

如果你暂时不想调用 API，可以改成：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=local
```

这会使用本地 hash embedding。效果不如真实 embedding，但适合学习和本地跑通。

## 为什么要保留本地 fallback

因为这个项目是学习版，每一步都应该能跑起来。如果没有 key、网络失败、千问接口失败，推荐系统不应该直接崩掉。

当前 fallback 规则：

```text
有真实 DASHSCOPE_API_KEY -> 使用千问 text-embedding-v4
没有 key / key 还是占位符 -> 使用本地 hash embedding
千问请求失败 -> ProductRecAgent 回退到规则召回
```

## 新增接口

查看向量库状态：

```http
GET /api/v1/vector-store
```

返回示例：

```json
{
  "backend": "chroma:local_hash:hashing-64",
  "collection": "products_local_hash_hashing_64_64",
  "persist_directory": "D:\\pycode\\agent\\cluade\\ecommerce-rebuild-step-by-step\\chroma_db",
  "indexed_fingerprint": null
}
```

## 当前推荐链路

```text
FastAPI
  -> Supervisor
  -> UserProfileAgent
  -> ProductRecAgent recall: Chroma
  -> ProductRecAgent rerank: rule score
  -> InventoryAgent
  -> MarketingCopyAgent
  -> ABTestEngine
  -> MetricsCollector
```

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
15 passed
```

## 下一步

Step 10 建议接入 Redis 实时特征窗口：

```text
SQLite -> 保存长期行为记录
Redis  -> 保存最近 1 小时 / 24 小时 / 7 天行为特征
```
