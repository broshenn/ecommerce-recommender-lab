# Step 6：SQLite 持久化

## 这个 Step 做了什么

把 Step 5 的内存行为事件存储升级成 SQLite 持久化。

之前：

```text
用户行为 -> Python 内存字典 -> 服务重启后丢失
```

现在：

```text
用户行为 -> SQLite user_events 表 -> 服务重启后仍可读取
```

## 新增内容

```text
app/database.py
data/app.sqlite3
```

`data/app.sqlite3` 是运行时生成的数据库文件，已经被 `.gitignore` 忽略，不会提交到仓库。

## 保持不变的接口

```text
POST /api/v1/events
GET /api/v1/users/{user_id}/events
GET /api/v1/users/{user_id}/profile
POST /api/v1/recommend
```

前端不需要知道底层从内存换成了 SQLite。

## 重点阅读文件

```text
app/database.py
app/behavior.py
app/main.py
tests/test_recommender.py
```

## 学习重点

这一阶段要看懂：业务接口可以保持稳定，底层存储可以从内存替换成数据库。

这是从演示版走向业务系统的重要一步。
