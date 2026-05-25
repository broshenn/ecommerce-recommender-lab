# Step 10：Redis 在线画像缓存 + 实时行为窗口

## 这一步要解决什么

原项目里的 Redis 不是主数据库，而是实时特征库 Feature Store：

```text
Redis Sorted Set 存用户行为序列
滑动窗口计算 1h / 24h / 7d 实时特征
profile:{user_id} 缓存离线或数据库聚合画像
UserProfileAgent 读取 Redis 特征生成用户画像
```

所以本项目 Step 10 也按这个思路实现。

## SQLite 和 Redis 的分工

```text
SQLite = 长期历史事实库
Redis  = 在线画像缓存 + 实时行为窗口
```

SQLite 保存完整历史：

```text
user_events(event_id, user_id, product_id, event_type, created_at)
```

Redis 保存在线特征：

```text
profile:{user_id}
behavior:{user_id}:view
behavior:{user_id}:like
behavior:{user_id}:dislike
behavior:{user_id}:add_to_cart
```

## 写流程

用户行为发生时：

```text
1. 写 SQLite user_events
2. 删除 Redis profile:{user_id}
3. 写 Redis behavior:{user_id}:{event_type}
```

这对应黑马点评里常见的缓存一致性策略：

```text
更新数据库 -> 删除缓存
```

这里不直接更新 `profile:{user_id}`，因为 profile 是聚合结果，直接改缓存容易并发覆盖或漏字段。

## 读流程

UserProfileAgent 读取画像时：

```text
1. 先读 Redis profile:{user_id}
2. 命中则直接返回
3. miss 则从 SQLite 聚合 user_events
4. 写回 Redis profile:{user_id}
5. 同时读取 Redis 行为窗口，放入 AgentResult 的 feature_store 字段
```

## 滑动窗口特征

当前实现：

```text
view_count_1h
view_count_24h
like_count_24h
dislike_count_24h
add_to_cart_count_7d
recent_views
recent_likes
recent_dislikes
recent_cart_items
recent_categories
recent_brands
recent_tags
rfm
```

RFM 当前用加购行为近似购买行为：

```text
Recency   -> 最近一次加购距离现在多远
Frequency -> 7 天内加购次数
Monetary  -> 加购商品平均价格
```

## 新增文件

```text
app/services/feature_store.py
steps/step-10-redis-feature-store/README.md
```

## 修改文件

```text
app/behavior.py
app/agents/user_profile_agent.py
app/main.py
app/services/__init__.py
requirements.txt
.env.example
README.md
steps/README.md
tests/test_recommender.py
CODE_UPDATES.md
```

## 新增接口

查看某个用户的 Redis 在线特征：

```http
GET /api/v1/feature-store/u001
```

返回包括：

```text
Redis 状态
实时窗口特征
当前缓存画像
```

## Redis 不可用怎么办

如果 Redis 没启动：

```text
写行为：SQLite 仍然成功，Redis 写入跳过
读画像：从 SQLite 正常聚合
推荐：继续可用
```

也就是说 Redis 是增强层，不是系统唯一依赖。

## 运行

确保 Redis 在本机运行：

```text
redis://localhost:6379/0
```

本机 Redis 启动脚本：

```powershell
D:\redis\Redis-8.4.0-Windows-x64-msys2-with-Service\start.bat
```

启动项目：

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
16 passed
```

## 下一步

Step 11 建议做 LLM 营销文案 Agent：

```text
MarketingCopyAgent
  -> LLM 生成个性化文案
  -> 合规规则过滤
  -> LLM 失败时使用模板兜底
```
