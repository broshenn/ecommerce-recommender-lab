# 电商推荐系统重构版

这是从零重构的学习版项目。目标不是一次性做完所有能力，而是每次只增加一个功能，让你能按阶段看懂系统如何成长。

## 当前阶段：Step 5 当前用户行为采集

当前已经实现：

- 商品目录：从 `data/products_amazon_sample.csv` 读取
- Amazon Reviews 2023 商品元数据 1000 条样本
- 英文商品标题到规则化中文展示名的适配
- 商品图片、评分、评分人数、原始类目来源字段
- 当前用户画像：类目、品牌、标签、预算、最近浏览
- 当前用户行为事件：查看、喜欢、不喜欢、加购
- 根据行为事件自动聚合画像
- 个性化打分排序
- 库存过滤：无库存商品不会被推荐
- 库存提示：低库存、热销限购会展示给前端
- 简单前端页面
- 基础单元测试

暂时不包含：

- LLM
- Agent
- Chroma
- Redis
- A/B 测试
- 训练出来的排序权重

## 本阶段你要重点看懂

- `data/products_amazon_sample.csv`：项目当前使用的 1000 条 Amazon 商品元数据样本
- `scripts/import_amazon_products.py`：从 Hugging Face 流式读取 Amazon Reviews 2023 元数据并生成 CSV
- `app/catalog.py`：从 CSV 读取商品，而不是把商品写死在代码里
- `app/models.py`：商品、推荐请求、用户行为、用户画像模型
- `app/behavior.py`：记录当前用户行为，并把行为聚合成用户画像
- `app/personalization.py`：当前用户画像打分规则，推荐排序的核心
- `app/inventory.py`：把“库存是否可卖、是否限购、是否紧张”单独拆出来
- `app/recommender.py`：合并手动画像和行为画像，再过滤库存、打分排序
- `app/static/index.html`：前端展示画像输入、行为按钮、商品图、评分、分数、库存提示

## 当前用户行为接口

记录行为：

```http
POST /api/v1/events
```

```json
{
  "user_id": "u001",
  "product_id": "B07ZPSG8P5",
  "event_type": "like"
}
```

支持的行为：

```text
view
like
dislike
add_to_cart
```

查看用户事件：

```http
GET /api/v1/users/u001/events
```

查看聚合后的当前用户画像：

```http
GET /api/v1/users/u001/profile
```

行为到画像的规则：

```text
view        -> recent_views
like        -> preferred_categories, liked_brands, preferred_tags
add_to_cart -> preferred_categories, liked_brands, preferred_tags, cart_items
dislike     -> disliked_products
```

## 当前用户画像字段

推荐接口支持手动画像字段，同时也会合并后端记录的当前用户行为画像：

```json
{
  "user_id": "u001",
  "preferred_categories": ["手机", "电子数码"],
  "liked_brands": ["Bastmei", "Sharp"],
  "preferred_tags": ["手机配件", "办公"],
  "budget_min": 50,
  "budget_max": 500,
  "recent_views": ["B07ZPSG8P5"],
  "disliked_products": [],
  "num_items": 3
}
```

打分规则先保持简单：

```text
类目匹配 +40
品牌匹配 +25
标签匹配 每个 +10
价格符合预算 +20
价格超出预算 -20
评分越高加分越多
最近浏览过 -30
不喜欢商品 -100
无库存商品不参与推荐
```

## Amazon 数据字段映射

当前只取 Amazon Reviews 2023 item metadata 的一部分字段：

```text
parent_asin       -> product_id
title             -> source_name
规则化中文展示名    -> name
main_category     -> source_category
人工中文类目       -> category
store             -> brand
price             -> price，按约 1 USD = 7.2 CNY 换算后取整
images[0].large   -> image_url
average_rating    -> rating
rating_number     -> rating_count
人工补充           -> stock, tags
```

这样做的原因是：当前项目重点是“当前用户画像 + 商品推荐”，所以先接入商品元数据和当前用户行为，不急着接入其他用户的历史行为。

## 后续训练计划

第四步里的分数目前还是人工规则。训练权重这件事先存下来，等基础框架搭完、再参考原始项目 `D:\pycode\agent\cluade\multi-agent-ecommerce-system` 后再做。

以后可以这样做：

```text
Amazon Reviews 2023 review rows
rating >= 4        -> 正向偏好
rating <= 2        -> 负向偏好
verified_purchase  -> 更强购买信号
用户特征 + 商品特征 -> LogisticRegression / LightGBM 排序模型
模型预测概率       -> 替代现在写死的推荐分数
```

## Step 快照

这个项目是学习版，每个阶段都应该有单独说明，但 `steps/` 里不保存代码快照，只保存阶段 README。

代码历史交给 Git commit 管理。阶段说明保存在：

```text
steps/step-04-user-profile-ranking/README.md
steps/step-05-current-user-behavior/README.md
```

后续约定：

```text
开发 Step 6 -> 测试通过 -> 写 steps/step-06-xxx/README.md
开发 Step 7 -> 测试通过 -> 写 steps/step-07-xxx/README.md
```

## 运行

```powershell
cd D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

打开：

```text
http://127.0.0.1:8010/
```

## 测试

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```
