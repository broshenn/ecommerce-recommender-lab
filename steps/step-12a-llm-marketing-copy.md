# Step 12a: MarketingCopyAgent LLM 改造 — 执行手册

## 背景

当前 Step 11 已完成：`llm_client` 就绪，`UserProfileAgent` 产出 LLM 画像（segments + intent_summary + recommendation_hint + price_sensitivity + rfm_interpretation）。`Supervisor` 已提取 `llm_hint` 传给下游 context。

但 `MarketingCopyAgent` 仍然是固定模板 if-else：

```python
# 现在的逻辑 (app/agents/marketing_copy_agent.py:39-44)
def _copy_for_product(self, product, profile):
    if profile and product.category in profile.preferred_categories:
        return "根据你的浏览偏好，为你优先推荐..."   # 永远这两句话
    if product.stock <= 100:
        return "...库存紧张..."
    return "...为你精选，兼顾品质..."
```

**本步目标：让 LLM 根据用户分群生成个性化文案，同时保留规则 fallback。**

---

## 1. 先读这些文件（了解上下文）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `app/agents/marketing_copy_agent.py` | 当前 45 行实现（等下要改） |
| 2 | `app/orchestrator/supervisor.py:127-134` | 怎么调 marketing_copy_agent，传了什么参数 |
| 3 | `app/agents/user_profile_agent.py:8-22` | SYSTEM_PROMPT 风格 + llm_profile 的 5 个字段 |
| 4 | `app/models.py:91-93` | `MarketingCopy` 模型（product_id + text） |
| 5 | `tests/test_recommender.py:46-48` | 现有测试怎么 mock LLM |

---

## 2. 改造文件：`app/agents/marketing_copy_agent.py`

### 设计要点

- 根据 `llm_profile.segments` 的首个标签选 prompt 模板
- 调 `llm_client.chat_json()` 生成文案
- LLM 不可用时降级走原 `_copy_for_product` 规则逻辑
- 广告法敏感词过滤（参考原项目）
- 不改 Agent 的对外接口——入参还是 `kwargs`，出参还是 `AgentResult` 包 `copies`

### 完整代码

```python
from __future__ import annotations

import re
from typing import Any

from app.models import AgentResult, MarketingCopy, Product, UserProfile

from app.agents.base_agent import BaseAgent
from app.services import llm_client

# 5 套 prompt 模板 × 用户分群
SEGMENT_TEMPLATES = {
    "new_user": """你是电商营销文案专家。为新用户撰写欢迎+推荐文案。
风格要求：热情友好、突出新人专属优惠感、降低决策门槛。
每个商品生成一条文案（25-40字）。""",

    "high_value": """你是电商营销文案专家。为高价值VIP用户撰写推荐文案。
风格要求：品质感、尊享感、突出商品高端属性和品牌价值。
每个商品生成一条文案（25-40字）。""",

    "price_sensitive": """你是电商营销文案专家。为价格敏感用户撰写推荐文案。
风格要求：突出性价比、促销价格、限时优惠、省钱金额。
每个商品生成一条文案（25-40字）。""",

    "churn_risk": """你是电商营销文案专家。为即将流失的用户撰写召回文案。
风格要求：情感唤回、专属折扣、限时活动、制造紧迫感。
每个商品生成一条文案（25-40字）。""",

    "active": """你是电商营销文案专家。为活跃用户撰写推荐文案。
风格要求：突出商品亮点和使用场景，引发共鸣。
每个商品生成一条文案（25-40字）。""",

    "category_explorer": """你是电商营销文案专家。为探索型用户撰写推荐文案。
风格要求：突出商品新奇特点，激发探索欲和好奇心。
每个商品生成一条文案（25-40字）。""",

    "brand_loyal": """你是电商营销文案专家。为品牌忠实用户撰写推荐文案。
风格要求：强化品牌认同感，突出品牌生态和兼容优势。
每个商品生成一条文案（25-40字）。""",

    "_default": """你是电商营销文案专家。为用户撰写个性化推荐文案。
风格要求：结合商品特点，简洁有力，突出实用价值。
每个商品生成一条文案（25-40字）。""",
}

# 广告法敏感词
FORBIDDEN_WORDS = [
    "最好", "第一", "国家级", "全球首", "绝对", "100%",
    "永久", "万能", "祖传", "纯天然", "最便宜", "全网最低",
]

OUTPUT_INSTRUCTION = """
请以JSON数组格式输出,每个元素格式:
[{"product_id": "xxx", "copy": "文案内容（25-40字）"}]
只输出JSON,不要其他内容。"""


class MarketingCopyAgent(BaseAgent):
    """Generate personalized marketing copy via LLM, with rule fallback."""

    def __init__(self):
        super().__init__(name="marketing_copy", timeout=5.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        profile: UserProfile | None = kwargs.get("profile")
        llm_profile: dict[str, Any] = kwargs.get("llm_profile", {})

        if not products:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"copies": [], "template": "empty", "copy_count": 0},
                confidence=1.0,
            )

        # === LLM 路径 ===
        llm_copies = self._generate_via_llm(products, profile, llm_profile)
        if llm_copies is not None:
            checked_copies = [self._compliance_check(c) for c in llm_copies]
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "copies": checked_copies,
                    "template": "llm_" + self._pick_segment(llm_profile),
                    "copy_count": len(checked_copies),
                    "mode": "llm",
                },
                confidence=0.9,
            )

        # === 规则 fallback ===
        copies = [
            MarketingCopy(
                product_id=product.product_id,
                text=self._copy_for_product(product, profile),
            )
            for product in products
        ]
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "copies": [copy.model_dump(mode="json") for copy in copies],
                "template": "rule_based_fallback",
                "copy_count": len(copies),
                "mode": "rule_fallback",
            },
            confidence=0.6,
        )

    def _generate_via_llm(
        self,
        products: list[Product],
        profile: UserProfile | None,
        llm_profile: dict[str, Any],
    ) -> list[dict[str, str]] | None:
        """Try LLM copy generation. Returns None if unavailable."""
        segment = self._pick_segment(llm_profile)
        system_prompt = SEGMENT_TEMPLATES.get(segment, SEGMENT_TEMPLATES["_default"])
        intent = llm_profile.get("intent_summary", "")

        product_lines = []
        for p in products:
            product_lines.append(
                f"- ID:{p.product_id} 名称:{p.name} 类目:{p.category} "
                f"价格:¥{p.price} 品牌:{p.brand} 标签:{','.join(p.tags)}"
            )

        user_message = (
            f"用户画像: {intent}\n\n"
            f"商品列表:\n" + "\n".join(product_lines)
        )

        result = llm_client.chat_json(
            system_prompt=system_prompt + "\n\n" + OUTPUT_INSTRUCTION,
            user_message=user_message,
            default=None,
        )
        if result is None:
            return None
        if isinstance(result, list) and len(result) > 0:
            return result
        return None

    def _pick_segment(self, llm_profile: dict[str, Any]) -> str:
        segments: list[str] = llm_profile.get("segments", [])
        if not segments:
            return "active"
        for s in segments:
            if s in SEGMENT_TEMPLATES:
                return s
        return "active"

    # === 规则 fallback 方法（保留原逻辑） ===
    def _copy_for_product(self, product: Product, profile: UserProfile | None) -> str:
        if profile and product.category in profile.preferred_categories:
            return f"根据你的浏览偏好，为你优先推荐 {product.name}，库存和价格都已为你校验。"
        if product.stock <= 100:
            return f"{product.name} 当前库存紧张，适合尽快决策。"
        return f"{product.name} 为你精选，兼顾品质、价格与实用性。"

    # === 广告法合规校验 ===
    def _compliance_check(self, copy_item: dict[str, str]) -> dict[str, str]:
        text = copy_item.get("copy", "")
        for word in FORBIDDEN_WORDS:
            text = re.sub(re.escape(word), "***", text)
        copy_item["copy"] = text
        return copy_item

    # === fallback ===
    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        copies = [
            {
                "product_id": p.product_id,
                "copy": f"{p.name} 为你精选，兼顾品质与实用，适合当前浏览偏好。",
            }
            for p in products
        ]
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={
                "copies": copies,
                "template": "fallback",
                "copy_count": len(copies),
                "mode": "agent_fallback",
            },
            confidence=0.4,
        )
```

### import 说明

```python
from app.services import llm_client   # ← 新增，模块顶部导入
import re                              # ← 新增，合规校验用
```

不再需要 `from app.agents.base_agent import BaseAgent` 改路径，原来的 `from app.agents.base_agent import BaseAgent` 保持不变。

---

## 3. 微调：`app/orchestrator/supervisor.py` 行 128-131

现在 marketing_copy_agent 的 `_execute` 需要 `llm_profile`，但 supervisor 只传了 `profile`。需要多传一个参数：

```python
# 找到行 127-131，改成：
copy_result = self.marketing_copy_agent.run(
    products=final_products,
    profile=profile,
    llm_profile=profile_result.data.get("llm_profile", {}),  # ← 新增
)
```

注意 `profile_result` 就是 Phase 1 拿到的 `UserProfileAgent` 返回结果，已经有 `llm_profile` 字段。

---

## 4. 验证

### 4.1 不加 API key 跑测试（测试 LLM 降级）

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest tests/test_recommender.py -q
```

当前测试已经 mock 了 LLM（`LLM_API_KEY=""`），所以 MarketingCopyAgent **必然走 rule fallback**，输出 `mode: "rule_fallback"`。所有现有测试应该通过。

如果现有测试在 `agent_results["marketing_copy"]` 的断言上有任何问题，调对应的断言（比如检查 `data["mode"]` 而不是 `data["template"]`）。

### 4.2 加上 API key 试 LLM 路径

确认 `.env` 有真实 `DEEPSEEK_API_KEY`，启动服务：

```powershell
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test001","num_items":2,"preferred_categories":["耳机"]}'
```

检查返回里的 `marketing_copies`：
- 文案不再是固定模板三句话
- 不同 user_id（不同 `llm_profile.segments`）可能看到不同风格
- `agent_results.marketing_copy.data.mode` 应该是 `"llm"`

### 4.3 检查合规

看返回的文案里如果 LLM 不小心出了 "最好"、"第一" 会被替换为 `***`。

---

## 5. 改动清单

| 动作 | 文件 | 说明 |
|------|------|------|
| 改造 | `app/agents/marketing_copy_agent.py` | 全量重写，~120 行 |
| 微调 | `app/orchestrator/supervisor.py` 行 128-131 | 多传 `llm_profile` |
