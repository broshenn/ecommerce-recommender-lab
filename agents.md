---
name: Agent LLM 改造计划
description: 从 Step 11 开始，逐步将规则 Agent 改造为 LLM 驱动的智能 Agent
updated: 2026-05-25
---

# Agent LLM 改造路线

## 前提

- 保持同步代码（不引入 async/await）
- LLM 调用直接用 `openai.OpenAI`（不引入 LangChain）
- 每次改动只聚焦一个 Agent，改完验证后再推进
- 改动前跑 `pytest -q` 保证现有测试通过

---

## Step 11: LLM Client + UserProfileAgent

**目标**：创建统一的 LLM 调用模块，改造用户画像 Agent 为 LLM 驱动。

### 文件变更

| 动作 | 文件 | 说明 |
|------|------|------|
| 新增 | `app/services/llm_client.py` | 封装 `openai.OpenAI`，读 env 配置，提供 `chat()` 方法 |
| 改造 | `app/agents/user_profile_agent.py` | `_execute` 从拼 dict 改为调 LLM 分析 Redis+SQLite 数据 |
| 改造 | `app/orchestrator/supervisor.py` | 消费 LLM 生成的画像摘要，传给下游 Agent |
| 修改 | `requirements.txt` | 加 `openai` |

### LLM Client 设计

```python
# app/services/llm_client.py — 核心接口
class LLMClient:
    def chat(self, system_prompt: str, user_message: str, **kwargs) -> str: ...
    def chat_json(self, system_prompt: str, user_message: str, **kwargs) -> dict: ...
```

- 读 `DEEPSEEK_API_KEY`、`DEEPSEEK_API_BASE`、`DEEPSEEK_MODEL`
- `chat_json` 在 `chat` 基础上自动解析 JSON 响应

### UserProfileAgent Prompt 设计

输入：Redis 实时特征（`get_user_features`）+ SQLite 画像（`build_user_profile`）

输出：
```json
{
  "segments": ["active", "audio_enthusiast"],
  "intent_summary": "正在从手机品类向音频扩展，Sony偏好明显",
  "recommendation_hint": "优先推荐Sony耳机+Apple生态互补配件",
  "price_sensitivity": "medium",
  "rfm_interpretation": "近期高频活跃，加购频繁但消费力中等"
}
```

### 验证方式

- `POST /api/v1/recommend` 返回的 `agent_results.user_profile.data` 里出现 LLM 生成的画像
- `GET /api/v1/users/{user_id}/profile` 能看到分群和意图摘要
- 现有测试保持通过

---

## Step 12: LLM 重排 + LLM 文案

**目标**：将 ProductRecAgent 的重排和 MarketingCopyAgent 改造为 LLM 驱动。

### ProductRecAgent 改造

| 改动 | 说明 |
|------|------|
| `_rerank()` 增加 LLM 路径 | 用户画像 + 候选商品列表 → LLM 排序输出 product_id 列表 |
| 保留规则 fallback | LLM 不可用时降级为 `score_product` 规则排序 |

Prompt 结构参考原项目 `RERANK_PROMPT`：传入用户画像摘要 + 候选商品（id/名称/类目/价格/标签），让 LLM 按相关性排序。

### MarketingCopyAgent 改造

| 改动 | 说明 |
|------|------|
| 5 套 Prompt 模板 | 根据用户分群选择：新客/高价值/价格敏感/活跃/流失 |
| LLM 生成文案 | 商品信息 + 用户画像 → LLM → 个性化文案 |
| 广告法合规 | 过滤敏感词：最好/第一/国家级/绝对/100% 等 |

### 验证方式

- 推荐结果排序与规则版有差异（LLM 会考虑语义相关性）
- 文案不再是固定模板，不同分群看到不同风格
- A/B 测试 control（规则）vs treatment（LLM）对比效果

---

## Step 13: A/B 升级 + 会话记忆

**目标**：A/B 测试引擎接入 Thompson Sampling，加入会话级短期记忆。

### A/B 测试升级

| 改动 | 说明 |
|------|------|
| `ABTestEngine` 增加 Thompson Sampling | 记录每次推荐曝光/点击结果，Beta 分布动态分配流量 |
| `GET /api/v1/experiments/outcome` | 前端上报点击事件，反馈给 A/B 引擎 |

参考原项目 `ab_test.py` 的 `assign_thompson` + `record_outcome`。

### 会话记忆

| 改动 | 说明 |
|------|------|
| 新增 `app/services/memory.py` | 短期记忆：当前 session 内的交互历史摘要 |
| 在 `RecommendRequest` 中传递 `session_id` | 同一会话多次推荐共享上下文 |

用于场景：用户第一次推荐点了 dislike → 第二次推荐自动避开该类目 → 记忆随 session 结束清空。

---

## Step 14: LangGraph 编排

**目标**：将 Supervisor 的硬编码执行流程改为 LangGraph 状态图。

### 改造内容

| 动作 | 文件 | 说明 |
|------|------|------|
| 新增 | `app/orchestrator/graph.py` | LangGraph 状态图，节点=Agent，边=执行顺序 |
| 新增 | `GET /api/v1/recommend/graph` | 走 LangGraph 管线的推荐端点 |
| 保留 | `app/orchestrator/supervisor.py` | 原 Supervisor 不动，两套编排并存 |

### 图结构

```
[start] → init(A/B分桶)
        → fan_out → {user_profile, product_recall}  (并行)
        → merge  → {rerank, inventory}              (并行)
        → filter → marketing_copy
        → aggregate → [end]
```

### LangGraph 的优势（相对硬编码）

- 状态自动在节点间传递，不需要手动管理变量
- 方便后续增加条件分支（如：库存不足 → 自动补召回）
- 可视化状态流转，方便面试讲解
- 内置 Checkpoint，支持断点续跑

---

## 全局约定

1. **LLM 调用统一走 `app/services/llm_client.py`**，不允许各 Agent 自己创建 client
2. **所有 LLM 调用必须有 rule fallback**，LLM 挂了降级走规则逻辑
3. **每次改造一个 Agent**，改完跑 `pytest -q`，确认通过再推进下一步
4. **不改动前端**，只通过 API 返回的 JSON 验证效果
5. **不做 async 改造**，Agent 保持同步 `_execute()`，并行继续用 `ThreadPoolExecutor`
