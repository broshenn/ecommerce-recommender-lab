# Step 5：当前用户行为采集

## 这个 Step 做了什么

把用户画像从“手动填写”升级成“系统根据当前用户行为自动聚合”。

新增行为：

```text
view
like
dislike
add_to_cart
```

新增接口：

```text
POST /api/v1/events
GET /api/v1/users/{user_id}/events
GET /api/v1/users/{user_id}/profile
```

## 行为如何变成画像

```text
view        -> recent_views
like        -> preferred_categories, liked_brands, preferred_tags
add_to_cart -> preferred_categories, liked_brands, preferred_tags, cart_items
dislike     -> disliked_products
```

推荐流程变成：

```text
手动画像 + 行为画像 -> 库存过滤 -> 个性化打分 -> 排序返回
```

## 重点阅读文件

```text
app/models.py
app/behavior.py
app/personalization.py
app/recommender.py
app/main.py
app/static/index.html
tests/test_recommender.py
```

## 学习重点

这一阶段要看懂：真实推荐系统不是只靠用户手动填偏好，而是不断把用户行为沉淀成画像。

当前行为数据存在内存里，重启服务后会清空。下一步适合做 SQLite 持久化。
