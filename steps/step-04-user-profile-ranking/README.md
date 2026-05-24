# Step 4：当前用户画像与个性化排序

## 这个 Step 做了什么

把推荐从“只按类目筛选”升级成“根据当前用户画像打分排序”。

画像字段包括：

```text
preferred_categories
liked_brands
preferred_tags
budget_min
budget_max
recent_views
```

推荐流程：

```text
商品目录 -> 库存过滤 -> 用户画像打分 -> 排序返回
```

## 打分规则

```text
类目匹配 +40
品牌匹配 +25
标签匹配 每个 +10
价格符合预算 +20
价格超出预算 -20
评分越高加分越多
最近浏览过 -30
```

## 重点阅读文件

```text
app/models.py
app/personalization.py
app/recommender.py
app/static/index.html
tests/test_recommender.py
```

## 学习重点

这一阶段要看懂：推荐系统里“用户画像 + 商品特征 -> 分数 -> 排序”的基本形态。

这里的权重仍然是人工写死的，后面会先记录行为数据，等基础框架完整后再考虑用训练模型替代手写权重。
