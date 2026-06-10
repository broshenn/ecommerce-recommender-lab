# IntelliCommerce Agent

面向电商场景的个性化推荐 Agent 系统。项目从 0 开始重构一个可学习、可演示、可写进简历的推荐系统实验室，核心目标不是只做一个“能返回商品列表”的接口，而是把真实推荐链路拆成多个可观测、可降级、可实验的 Agent：用户画像、商品召回、智能重排、库存过滤、营销文案生成和 A/B 实验闭环。

项目当前默认推荐入口是：

```http
POST /api/v1/recommend/graph
```

也就是 LangGraph 编排版本。

---

## 1. 项目一句话

> 基于 FastAPI + Redis + SQLite + Chroma + LangGraph + LLM 构建电商个性化推荐 Agent 系统，实现用户实时画像、向量召回、LLM 重排、库存过滤、个性化营销文案生成、LoRA 文案模型微调与 A/B 实验闭环。

这个项目适合用来学习三类能力：

| 学习方向 | 项目中对应能力 |
| --- | --- |
| 推荐系统工程 | 用户画像、召回、重排、库存过滤、曝光与点击闭环 |
| Agent 系统设计 | 多 Agent 拆分、状态流编排、fallback、可观测性 |
| LLM 应用落地 | LLM 画像、LLM rerank、LLM copy、LoRA 微调、本地推理 |

---

## 2. 技术栈

| 层级 | 技术 |
| --- | --- |
| Web 服务 | Python, FastAPI, Uvicorn |
| 前端 | Vue 3, 原生 HTML/CSS |
| 编排 | LangGraph, Supervisor Orchestrator |
| 数据存储 | SQLite, Redis |
| 向量召回 | Chroma, local hash embedding, DashScope/Qwen embedding |
| LLM 服务 | DeepSeek API, OpenAI-compatible API |
| 本地模型 | Qwen2.5-3B LoRA, Ollama GGUF, HuggingFace safetensors |
| 实验评估 | A/B test, exposure/outcome, CTR, Thompson Sampling 预留 |
| 测试 | pytest |

---

## 3. 系统全景

### 3.1 为什么要拆成多 Agent

电商推荐不是一个单点问题。一个真实请求通常同时包含这些问题：

1. 用户是谁，最近是否活跃，长期喜欢什么？
2. 商品池里哪些商品可能相关？
3. 哪些商品应该排在前面？
4. 商品是否有库存，是否需要限购或提示低库存？
5. 如何给不同用户写不同风格的推荐文案？
6. LLM 增强链路到底比规则链路好不好？

所以本项目把推荐链路拆成多个 Agent，每个 Agent 只负责一个清晰的阶段。

```text
用户请求
  -> UserProfileAgent      用户画像 Agent
  -> ProductRecAgent       商品召回 / 重排 Agent
  -> InventoryAgent        库存过滤 Agent
  -> MarketingCopyAgent    营销文案 Agent
  -> ABTestEngine          实验分流与结果记录
  -> 推荐结果返回前端
```

### 3.2 LangGraph 编排链路

当前主链路由 `app/orchestrator/graph.py` 编排：

```text
init
  -> phase1
       并行执行 UserProfileAgent + ProductRecAgent recall
  -> merge1
       把 llm_profile.recommendation_hint 写入 effective_request.context.llm_hint
  -> phase2
       并行执行 ProductRecAgent rerank + InventoryAgent
  -> merge2
       合并重排结果和库存结果
  -> expand
       如果商品不足，扩大召回范围后重新进入 phase2
  -> phase3
       MarketingCopyAgent 生成个性化文案
  -> aggregate
       记录指标、实验曝光，返回 RecommendResponse
```

对应接口：

```http
POST /api/v1/recommend/graph
```

旧版 Supervisor 链路仍然保留：

```http
POST /api/v1/recommend
```

---

## 4. 核心模块

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| FastAPI 入口 | `app/main.py` | 路由注册、静态页面、推荐接口、实验接口 |
| LangGraph 编排 | `app/orchestrator/graph.py` | 当前默认推荐链路 |
| Supervisor 编排 | `app/orchestrator/supervisor.py` | 早期编排链路，保留用于对比 |
| 数据模型 | `app/models.py` | 请求、响应、商品、画像、实验模型 |
| 商品目录 | `app/catalog.py` | 读取 Amazon 商品样本 CSV |
| 行为系统 | `app/behavior.py` | view/like/dislike/purchase 记录与画像聚合 |
| 规则打分 | `app/personalization.py` | 类目、品牌、标签、预算等规则评分 |
| 库存逻辑 | `app/inventory.py` | 库存状态、低库存、限购提示 |
| 用户画像 Agent | `app/agents/user_profile_agent.py` | SQLite + Redis + LLM 生成用户画像 |
| 推荐 Agent | `app/agents/product_rec_agent.py` | Chroma 召回、规则重排、LLM 重排 |
| 库存 Agent | `app/agents/inventory_agent.py` | 过滤无库存商品，返回库存提示 |
| 文案 Agent | `app/agents/marketing_copy_agent.py` | 规则文案、LLM 文案、本地 LoRA 文案 |
| LLM 客户端 | `app/services/llm_client.py` | OpenAI-compatible Chat API 统一封装 |
| 本地 HF 文案模型 | `app/services/local_hf_copy_client.py` | HuggingFace 本地 GPU 推理封装 |
| Redis 特征 | `app/services/feature_store.py` | 在线画像缓存和实时行为窗口 |
| 向量库 | `app/services/vector_store.py` | Chroma 商品向量召回 |
| A/B 实验 | `app/services/ab_test.py` | control/treatment 分流、曝光、点击统计 |
| 指标 | `app/services/metrics.py` | Agent latency、业务事件统计 |
| 前端 | `app/static/index.html` | Vue 3 推荐演示页面 |

---

## 5. 数据与画像

### 5.1 商品数据

商品目录来自 Amazon Reviews 2023 的抽样整理数据，当前约 1000 条商品。

核心字段包括：

```text
product_id
name
category
brand
price
stock
tags
image_url
rating
rating_count
source_dataset
```

文件位置：

```text
data/products_amazon_sample.csv
```

### 5.2 用户行为

前端按钮会记录用户行为：

| 行为 | 含义 |
| --- | --- |
| `view` | 浏览商品 |
| `like` | 喜欢商品 |
| `dislike` | 不喜欢商品 |
| `purchase` | 购买 |

行为写入 SQLite，作为长期画像的事实来源。

### 5.3 SQLite 长期画像

SQLite 保存用户历史行为。系统会从历史行为中聚合：

```text
preferred_categories
liked_brands
preferred_tags
recent_views
disliked_products
cart_items
event_count
```

直观理解：

```text
SQLite = 用户长期事实库
```

它适合保存“已经发生过、不能丢”的行为事实。

### 5.4 Redis 实时特征

Redis 用于保存短期在线特征和缓存画像：

```text
profile:{user_id}
behavior:{user_id}:view
behavior:{user_id}:like
behavior:{user_id}:dislike
behavior:{user_id}:purchase
```

在线特征包括：

```text
最近 1 小时浏览次数
最近 24 小时浏览次数
最近 24 小时点赞次数
最近 24 小时不喜欢次数
最近 7 天购买次数
最近兴趣类目
最近兴趣品牌
最近兴趣标签
RFM 指标
```

直观理解：

```text
Redis = 在线实时特征层
```

它适合保存“最近刚发生、会快速变化”的行为信号。

---

## 6. 推荐链路详解

### 6.1 UserProfileAgent：用户画像 Agent

`UserProfileAgent` 做三件事：

1. 从 SQLite 读取长期行为画像。
2. 从 Redis 读取实时行为窗口。
3. 调用 LLM 生成更高层的用户理解。

LLM 输出结构：

```json
{
  "segments": ["new_user", "price_sensitive"],
  "intent_summary": "用户近期关注手机配件，偏好价格适中的保护壳",
  "recommendation_hint": "优先推荐高评分、价格友好的手机保护类商品",
  "price_sensitivity": "high",
  "rfm_interpretation": "近期活跃度较低，但存在明确的品类偏好"
}
```

这些字段会继续影响后续重排和文案生成。

### 6.2 ProductRecAgent：商品召回与重排

推荐系统通常分成两层：

```text
召回 = 从大商品池中快速找出可能相关的一批候选
重排 = 对候选商品做更精细的排序
```

本项目的召回层：

```text
Chroma 向量召回
  -> 可用 DashScope/Qwen embedding
  -> 也可降级到本地 hash embedding
```

本项目的重排层：

```text
如果存在 llm_hint:
    尝试 LLM rerank
    失败则回退规则打分
否则:
    使用规则打分
```

规则打分综合考虑：

```text
类目匹配
品牌匹配
标签匹配
预算区间
最近浏览
不喜欢商品过滤
评分和评分数量
```

### 6.3 InventoryAgent：库存过滤 Agent

库存 Agent 负责把推荐结果从“相关”变成“可买”。

它会处理：

```text
无库存过滤
低库存提示
限购提示
库存状态标注
```

示例返回字段：

```json
{
  "stock_status": "low",
  "stock_message": "库存紧张，建议尽快决策",
  "purchase_limit": 1
}
```

### 6.4 MarketingCopyAgent：营销文案 Agent

文案 Agent 根据用户分群和商品信息生成推荐语。

当前链路：

```text
control 组:
  规则模板文案

treatment 组:
  优先本地 LoRA 文案模型 / Ollama
  失败后降级 DeepSeek API
  再失败后降级规则模板
```

针对不同用户分群，文案风格不同：

| 分群 | 文案倾向 |
| --- | --- |
| `new_user` | 友好、低门槛、欢迎首次探索 |
| `active` | 强调使用场景和商品亮点 |
| `high_value` | 品质感、品牌感、体验价值 |
| `price_sensitive` | 性价比、真实价格、预算友好 |
| `churn_risk` | 温和召回、品质提醒、避免虚假紧迫感 |
| `category_explorer` | 探索感、新奇感、多样选择 |
| `brand_loyal` | 品牌认同、系列搭配、兼容生态 |

---

## 7. LLM 与 LoRA 微调

### 7.1 为什么要微调文案模型

原始小模型在电商文案任务上容易出现三个问题：

| 问题 | 表现 |
| --- | --- |
| 价格幻觉 | 编造折扣、满减、限时优惠 |
| 长度不稳定 | 要求 25-40 字，但经常过短 |
| 风格不清晰 | 不同用户分群生成的文案差异不明显 |

这些问题非常适合通过“高质量样本 + LoRA”进行针对性修正。

### 7.2 训练数据

当前文案 LoRA 使用约 940 条样本。

数据构造目标：

```text
商品信息 + 用户分群 + 安全约束 -> 25-40 字中文营销文案
```

重点强化：

```text
price_sensitive 不编造折扣
churn_risk 不制造虚假紧迫感
new_user 不只写泛泛欢迎
brand_loyal 突出品牌一致性
active 和 high_value 拉开风格
```

### 7.3 训练框架

训练框架使用 LLaMA-Factory。

基座模型：

```text
Qwen2.5-3B
```

训练产物：

```text
LoRA adapter
合并后的 HuggingFace safetensors 模型
Ollama 可导入的 GGUF 模型
```

本地 HuggingFace 模型默认路径：

```text
D:\models\ecom-copy-lora-merged
```

### 7.4 本地推理策略

项目中支持两种本地文案模型路线：

| 路线 | 优点 | 注意点 |
| --- | --- | --- |
| Ollama GGUF | 启动简单、速度快、资源占用低 | JSON 输出偶发需要容错 |
| HuggingFace GPU | 更贴近训练后原始模型效果 | 需要 CUDA PyTorch 和显存 |

当前代码里 `MarketingCopyAgent` 支持本地模型优先、DeepSeek API 降级的思路。

HuggingFace 本地文案客户端：

```text
app/services/local_hf_copy_client.py
```

---

## 8. A/B 实验闭环

### 8.1 control 与 treatment

系统内置 A/B 实验：

```text
control   -> 规则画像 + 规则重排 + 规则文案
treatment -> LLM 画像 + LLM 重排 + LLM 文案
```

分组由稳定 hash 完成，同一个 `user_id` 会稳定进入同一组。

### 8.2 实验数据

每次推荐成功后记录曝光：

```text
record_exposure()
```

用户点击、喜欢、购买、不喜欢后回传 outcome：

```http
POST /api/v1/experiments/{experiment_id}/outcome
```

统计字段包括：

```text
exposures
clicks
skips
ctr
expected_ctr
alpha
beta
```

这使项目不只是“调用 LLM 生成结果”，而是可以继续回答：

```text
LLM 文案是否真的带来更高点击率？
LLM 重排是否真的优于规则排序？
本地微调模型是否值得替代付费 API？
```

---

## 9. 前端演示

前端位于：

```text
app/static/index.html
```

当前页面支持：

```text
填写 user_id
选择推荐数量
选择偏好类目 / 品牌 / 标签
设置预算
发起 LangGraph 推荐
展示商品图片、价格、库存、评分、推荐理由
展示营销文案
记录 view / like / dislike / purchase
回传 A/B outcome
展示 Agent 调试信息
```

访问地址：

```text
http://127.0.0.1:8010/
```

---

## 10. 快速开始

### 10.1 克隆项目

```powershell
git clone https://github.com/broshenn/ecommerce-recommender-lab.git
cd ecommerce-recommender-lab
```

### 10.2 安装依赖

```powershell
D:\anaconda\envs\py3.10\python.exe -m pip install -r requirements.txt
```

如果要在本地用 HuggingFace GPU 跑文案模型，需要额外安装 CUDA 版 PyTorch。示例：

```powershell
D:\anaconda\envs\py3.10\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
D:\anaconda\envs\py3.10\python.exe -m pip install --upgrade "transformers>=4.40.0" accelerate safetensors
```

具体 CUDA 版本请根据本机显卡驱动选择。

### 10.3 配置环境变量

复制模板：

```powershell
copy .env.example .env
```

基础配置：

```env
REDIS_URL=redis://localhost:6379/0
FEATURE_STORE_PROFILE_TTL_SECONDS=600
FEATURE_STORE_BEHAVIOR_TTL_SECONDS=604800
```

DeepSeek 配置：

```env
DEEPSEEK_API_KEY=你的DeepSeekKey
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.3
LLM_TIMEOUT_SECONDS=15
```

Chroma embedding 配置：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=local
```

如果要使用 DashScope/Qwen embedding：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=qwen
DASHSCOPE_API_KEY=你的DashScopeKey
```

本地 HuggingFace 文案模型配置：

```env
COPY_LLM_BACKEND=hf
COPY_LLM_HF_MODEL_PATH=D:\models\ecom-copy-lora-merged
COPY_LLM_HF_DEVICE=cuda
COPY_LLM_HF_MAX_NEW_TOKENS=768
```

Ollama 文案模型配置：

```env
OLLAMA_COPY_MODEL=ecom-copy-lora:qwen25-3b-gguf
OLLAMA_COPY_BASE_URL=http://127.0.0.1:11434/v1
```

`.env` 已被 `.gitignore` 忽略，不会提交到 GitHub。

### 10.4 启动 Redis

本地 Windows 示例：

```powershell
D:\redis\Redis-8.4.0-Windows-x64-msys2-with-Service\start.bat
```

如果 Redis 没启动，系统会尽量降级，但实时画像能力会受影响。

### 10.5 启动服务

```powershell
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

打开：

```text
http://127.0.0.1:8010/
```

---

## 11. 常用接口

### 11.1 健康检查

```http
GET /health
```

### 11.2 LangGraph 推荐

```http
POST /api/v1/recommend/graph
```

示例请求：

```json
{
  "user_id": "u001",
  "scene": "homepage",
  "num_items": 5,
  "preferred_categories": ["手机", "电子数码"],
  "liked_brands": ["Bastmei", "Sharp"],
  "preferred_tags": ["手机配件", "办公", "保护壳"],
  "budget_min": 0,
  "budget_max": 300,
  "recent_views": [],
  "disliked_products": [],
  "context": {}
}
```

响应中重点看：

```text
products
marketing_copies
experiment_group
agent_results.user_profile
agent_results.product_recall
agent_results.product_rerank
agent_results.inventory
agent_results.marketing_copy
```

### 11.3 记录用户行为

```http
POST /api/v1/events
```

示例：

```json
{
  "user_id": "u001",
  "product_id": "B07ZPSG8P5",
  "event_type": "like"
}
```

### 11.4 查看用户画像

```http
GET /api/v1/users/u001/profile
```

### 11.5 查看 Redis 在线特征

```http
GET /api/v1/feature-store/u001
```

### 11.6 查看 A/B 实验

```http
GET /api/v1/experiments?user_id=u001
```

### 11.7 回传实验 outcome

```http
POST /api/v1/experiments/recommendation_strategy_v1/outcome
```

示例：

```json
{
  "experiment_id": "recommendation_strategy_v1",
  "group": "treatment",
  "user_id": "u001",
  "success": true,
  "product_id": "B07ZPSG8P5"
}
```

### 11.8 查看向量库状态

```http
GET /api/v1/vector-store
```

### 11.9 查看指标

```http
GET /api/v1/metrics
```

---

## 12. 测试

运行单元测试：

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

语法检查：

```powershell
D:\anaconda\envs\py3.10\python.exe -m compileall app tests
```

---

## 13. 离线推荐评测

没有真实线上流量时，项目先用 Amazon 商品数据生成 weak-label 用户行为样本，用来验证推荐链路是否命中用户未来兴趣意图。

生成行为样本：

```powershell
D:\anaconda\envs\py3.10\python.exe scripts\import_amazon_user_events.py --generate-sample --max-events 500 --output data\amazon_user_events_sample.csv
```

运行端到端离线评测：

```powershell
D:\anaconda\envs\py3.10\python.exe scripts\evaluate_recommendation_offline.py --events data\amazon_user_events_sample.csv --orchestrator graph --k 5 --max-users 50
```

核心指标：

```text
Exact Hit@K: 是否命中未来同一个商品 ASIN，最严格。
Intent Hit@K: 是否命中未来正反馈商品的类目、品牌、标签意图。
Budget Compliance: 推荐商品是否满足预算约束。
Inventory Compliance: 推荐商品是否有库存。
Avg Latency: 端到端推荐耗时。
Fallback Rate: 文案 Agent 是否降级到规则兜底。
```

---

## 14. 如何判断 LLM 链路是否生效

前端或 API 响应里重点看：

```text
experiment_group
agent_results.user_profile.data.llm_client.available
agent_results.user_profile.data.llm_profile.recommendation_hint
agent_results.product_rerank.data.mode
agent_results.marketing_copy.data.mode
agent_results.marketing_copy.data.template
```

常见情况：

| 现象 | 说明 |
| --- | --- |
| `experiment_group=control` | 正常走规则链路 |
| `experiment_group=treatment` 且 `product_rerank.mode=llm_rerank` | LLM 重排生效 |
| `marketing_copy.mode=llm` | LLM 或本地文案模型生效 |
| `marketing_copy.mode=rule_fallback` | 文案模型和 API 都失败，回退规则 |
| `llm_client.available=false` | API key 未配置或不可用 |

如果只是想强行测试 treatment，可以换一个不同的 `user_id`，因为分组由稳定 hash 决定。

---

## 15. 学习路线

建议按下面顺序读源码：

| 顺序 | 文件 | 重点 |
| --- | --- | --- |
| 1 | `app/models.py` | 先看系统输入输出长什么样 |
| 2 | `app/main.py` | 看接口如何暴露 |
| 3 | `app/orchestrator/graph.py` | 看 LangGraph 如何串联 Agent |
| 4 | `app/agents/user_profile_agent.py` | 看 SQLite、Redis、LLM 如何合成画像 |
| 5 | `app/agents/product_rec_agent.py` | 看召回、重排、LLM rerank 和 fallback |
| 6 | `app/agents/inventory_agent.py` | 看库存过滤如何独立成 Agent |
| 7 | `app/agents/marketing_copy_agent.py` | 看文案 Agent、分群 prompt、本地模型降级链路 |
| 8 | `app/services/ab_test.py` | 看实验分流和结果统计 |
| 9 | `app/static/index.html` | 看前端如何触发推荐和回传行为 |
| 10 | `training/` | 看 LoRA 数据准备与训练配置 |

配套阶段记录：

```text
CODE_UPDATES.md
steps/
```

---

## 16. 项目亮点

### 15.1 工程亮点

```text
多 Agent 拆分
LangGraph 状态图编排
SQLite + Redis 双层画像
Chroma 向量召回
LLM rerank + 规则 fallback
LLM copy + 本地 LoRA 模型
A/B 实验闭环
前端调试面板
Agent latency 可观测
```

### 15.2 推荐系统亮点

项目覆盖了推荐系统的完整骨架：

```text
行为采集
画像构建
候选召回
个性化重排
库存约束
结果展示
曝光记录
点击/购买回传
实验统计
```

### 15.3 LLM 应用亮点

项目不是简单调用一次 LLM，而是把 LLM 放在三个不同位置：

| 位置 | 作用 |
| --- | --- |
| 用户画像 | 从长期行为和实时特征中总结用户意图 |
| 商品重排 | 根据 recommendation hint 对候选商品排序 |
| 营销文案 | 根据用户分群生成个性化推荐语 |

同时对文案模型做了 LoRA 微调，用来解决价格幻觉和风格分化问题。

---

## 17. 简历描述参考

```text
IntelliCommerce Agent：智能电商推荐与营销生成系统

基于 FastAPI + LangGraph + Redis + SQLite + Chroma + Qwen2.5 LoRA 构建电商个性化推荐 Agent 系统，设计用户画像、商品召回/重排、库存过滤、营销文案生成与 A/B 实验闭环；使用 SQLite 沉淀长期行为、Redis 构建实时特征、Chroma 完成向量召回，并结合 DeepSeek API 与本地 LoRA 文案模型实现 LLM 画像、LLM 重排和个性化营销文案生成。
```

简历 bullet 版本：

```text
- 设计 UserProfileAgent / ProductRecAgent / InventoryAgent / MarketingCopyAgent，并基于 LangGraph 编排推荐链路，实现 Agent 状态流转、异常兜底和结果聚合。
- 使用 SQLite 持久化用户行为，结合 Redis 构建实时画像、行为窗口和 RFM 特征，支持 LLM 生成用户分群、购买意图和推荐提示。
- 基于 Amazon Reviews 2023 构建约 1000 条商品目录，接入 Chroma 向量召回，并融合规则评分与 LLM recommendation hint 完成个性化重排。
- 构造 940 条高质量文案样本，使用 LLaMA-Factory 对 Qwen2.5-3B 进行 LoRA 微调，缓解价格幻觉、文案过短和分群风格不明显问题。
- 实现 control/treatment 分流、曝光记录、点击/购买 outcome 回传和实验统计，为评估 LLM 重排与 LLM 文案效果提供数据闭环。
```

---

## 18. 后续规划

| 优先级 | 方向 | 目标 |
| --- | --- | --- |
| P0 | 前端调试面板完善 | 更清楚展示每个 Agent 的耗时、模式和 fallback 原因 |
| P0 | 推荐日志沉淀 | 保存曝光、点击、购买、分组、商品 ID、画像快照 |
| P1 | 实验报告接口 | 输出 control/treatment 的曝光、点击、CTR、购买率 |
| P1 | 文案模型评测集 | 固定 case 评估 JSON、幻觉、长度、风格分化 |
| P2 | 画像模型蒸馏 | 当 prompt 修复不足时，再考虑蒸馏 UserProfileAgent |
| P2 | 排序模型训练 | 基于真实 outcome 训练轻量排序模型 |

---

## 19. 参考资料

- FastAPI: https://fastapi.tiangolo.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Chroma: https://docs.trychroma.com/
- Redis: https://redis.io/docs/latest/
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Qwen: https://github.com/QwenLM/Qwen
- Amazon Reviews 2023: https://amazon-reviews-2023.github.io/
