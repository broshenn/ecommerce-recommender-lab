# Step 13a: A/B 实验驱动策略开关

这一阶段把之前“只是分桶展示”的 A/B 实验，升级成真正控制推荐策略的开关。

## 这一步解决什么问题

Step 12b 之后，项目已经有三条 LLM 能力：

```text
UserProfileAgent    -> LLM 用户画像
ProductRecAgent     -> LLM 商品重排
MarketingCopyAgent  -> LLM 推荐文案
```

但之前的问题是：只要 `.env` 里配置了 LLM，所有用户都有机会走 LLM。A/B 实验虽然返回了 `control` / `treatment`，但没有真正影响执行路径。

这一步改成：

```text
control   -> 纯规则链路：规则画像 + 规则重排 + 规则文案
treatment -> LLM 增强链路：LLM画像 + LLM重排 + LLM文案
```

## 改了哪些代码

```text
app/services/ab_test.py
```

把实验配置改成真实策略配置：

```text
control.config.strategy   = rule
treatment.config.strategy = llm
```

并且分别声明 `profile`、`rerank`、`copy` 三个子策略，方便前端或后续指标分析读取。

```text
app/orchestrator/supervisor.py
```

Supervisor 现在会把 `experiment.group` 传给 Agent。

关键逻辑：

```text
Phase 1:
  UserProfileAgent(request, experiment_group)

Phase 2:
  只有 treatment 才把 llm_profile.recommendation_hint 写进 request.context.llm_hint

Phase 3:
  MarketingCopyAgent(products, profile, llm_profile, experiment_group)
```

```text
app/agents/user_profile_agent.py
```

control 组不调用 LLM，直接返回规则画像占位结果，并且 `recommendation_hint` 保持空字符串。

这样后面的 `ProductRecAgent` 看不到 `llm_hint`，自然会走 `score_product()` 规则重排。

```text
app/agents/marketing_copy_agent.py
```

control 组不调用 LLM，直接走 `_copy_for_product()` 模板文案，返回：

```text
mode = control_rule
```

## 执行路径对比

| 阶段 | control | treatment |
|---|---|---|
| 用户画像 | SQLite + Redis 规则画像 | SQLite + Redis + LLM 画像 |
| 重排 | `score_product()` | 优先 LLM rerank，失败回退规则 |
| 文案 | `_copy_for_product()` | 优先 LLM copy，失败回退规则 |
| LLM 调用 | 0 次 | 最多 3 次 |
| 适合作用 | 对照组、稳定基线 | 新策略实验组 |

## 怎么验证

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

当前验证结果：

```text
19 passed
```

你也可以查两个固定用户：

```powershell
curl "http://127.0.0.1:8010/api/v1/experiments?user_id=ab-user-1"
curl "http://127.0.0.1:8010/api/v1/experiments?user_id=ab-user-2"
```

当前稳定分桶：

```text
ab-user-1 -> control
ab-user-2 -> treatment
```

## 你阅读时抓住一个点

这一步的重点不是“新增推荐算法”，而是“让实验配置影响执行链路”。

后面如果要做 Thompson Sampling、灰度发布、按用户分层实验，本质上都是继续增强这个策略开关。
