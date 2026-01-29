# Hidden Liquidity Inference Framework for DWTS Fan Vote Estimation

> **核心思想**：将选秀节目建模为**双边信息不对称市场**，评委打分是"公开报价"，粉丝投票是"暗池订单"。通过分析选手的"抗跌性"——即低分时的存活概率——逆向破解隐藏的粉丝投票分布。

---

## 一、问题建模：市场微观结构类比

### 1.1 核心对应关系

| DWTS 选秀竞赛 | 金融市场类比 | 数学符号 |
|---------------|--------------|----------|
| 选手 $i$ 在第 $w$ 周 | 资产 $i$ 在交易日 $t$ | $(i, w)$ |
| 评委总分 $J_{i,w}$ | 公开报价/市价订单 | Observable |
| 粉丝投票 $V_{i,w}$ | 暗池订单/隐藏限价单 | **Latent (需推断)** |
| 被淘汰 | 被"清仓"出局 | $E_{i,w} = 1$ |
| 存活晋级 | 持仓保留 | $S_{i,w} = 1$ |
| 综合排名 | 综合账面价值 | $R_{i,w}$ |

### 1.2 核心假设

**假设 H1 (理性淘汰)**：每周综合得分最低的选手被淘汰。

**假设 H2 (隐藏流动性)**：粉丝投票 $V_{i,w}$ 不可直接观测，但其效应体现在选手的"存活韧性"中。

**假设 H3 (机制已知)**：
- **Rank-Based (S1-2, S28-34)**：$R_{i,w} = \text{JudgeRank}_i + \text{FanRank}_i$
- **Percent-Based (S3-27)**：$R_{i,w} = \frac{J_{i,w}}{\sum_j J_{j,w}} + \frac{V_{i,w}}{\sum_j V_{j,w}}$

---

## 二、算法核心：隐藏流动性的区间推断

### 2.1 阶段一：构建约束系统 (Constraint System)

对于每周的淘汰事件，我们可以建立**不等式约束**：

**定义**：设第 $w$ 周有 $n_w$ 名选手参赛，淘汰者为 $e_w$。

**约束**：对于所有存活者 $i \neq e_w$：
$$
\text{CombinedScore}(i, w) > \text{CombinedScore}(e_w, w)
$$

展开（以 Rank-Based 为例）：
$$
\text{JudgeRank}_i + \text{FanRank}_i < \text{JudgeRank}_{e_w} + \text{FanRank}_{e_w}
$$

由于 $V_{i,w}$ 未知，但 Rank 是 $V$ 的函数，这构成了一个**关于粉丝票序的约束系统**。

### 2.2 阶段二：可行域枚举与蒙特卡洛采样

**目标**：找到所有满足约束的 $(V_{1,w}, V_{2,w}, ..., V_{n_w,w})$ 排列组合。

**方法 A：排列约束求解 (Exact but Exponential)**
```
For each week w:
    1. 获取评委分排名 JudgeRank[1..n]
    2. 枚举所有可能的 FanRank 排列 Π
    3. 对每个 Π, 计算 CombinedRank
    4. 保留使淘汰者 e_w 在 CombinedRank 中排最后的 Π
    5. 这些 Π 构成"可行粉丝投票序集"
```

**方法 B：逆向贝叶斯采样 (Scalable)**
```
1. 假设 V ~ Prior(选手属性, 历史表现)
2. 使用 MCMC/ABC 采样，接受满足淘汰约束的样本
3. 得到 V 的后验分布
```

### 2.3 阶段三：流动性指标构建

定义选手 $i$ 的**隐藏流动性深度 (Hidden Liquidity Depth, HLD)**：

$$
\text{HLD}_{i,w} = \mathbb{E}[\text{FanRank}_i \mid \text{Constraints Satisfied}]
$$

解释：
- HLD 越小 → 粉丝票排名越靠前 → 隐藏支持越强
- HLD 越大 → 主要靠评委分存活

**极端案例诊断**：
| 选手类型 | 评委分排名 | HLD | 诊断 |
|----------|------------|-----|------|
| Bristol Palin | 几乎垫底 (12次) | 极小 | 超强铁粉群体 |
| 技术型冠军 | 顶尖 | 中等 | 评委与粉丝一致 |
| "冷门"淘汰 | 中游 | 极大 | 粉丝基础薄弱 |

---

## 三、分层模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: 争议分析                         │
│  - 针对 Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby   │
│    Bones 等争议选手，进行流动性压力测试                        │
│  - 对比 Rank vs Percent 机制下的结果差异                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Layer 2: 粉丝票推断                        │
│  - 输入: 每周评委分、存活/淘汰标签、选手属性                    │
│  - 方法: 约束满足 + MCMC 采样                                  │
│  - 输出: V_{i,w} 的区间估计 [V_low, V_high]                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Layer 1: 数据预处理                        │
│  - 解析评委分（处理 N/A, 多舞平均, bonus 分）                   │
│  - 提取淘汰事件序列                                            │
│  - 识别投票机制切换点 (S2→S3, S27→S28)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、模型 M1：基于约束的排列推断

### 4.1 输入
- $J_{i,w}$: 选手 $i$ 在第 $w$ 周的评委总分
- $E_w$: 第 $w$ 周被淘汰选手的 ID
- 投票机制 $M \in \{\text{Rank}, \text{Percent}\}$

### 4.2 算法伪代码

```python
def infer_fan_votes_week(judge_scores, eliminated_id, mechanism):
    """
    推断单周粉丝投票的可行排列
    
    Returns:
        feasible_fan_rankings: List of all valid fan vote orderings
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    judge_ranks = compute_ranks(judge_scores)  # 越高分排名越小(好)
    
    feasible = []
    
    # 枚举所有 n! 粉丝排名排列
    for fan_ranks in permutations(range(1, n+1)):
        fan_rank_dict = dict(zip(contestants, fan_ranks))
        
        if mechanism == "Rank":
            combined = {c: judge_ranks[c] + fan_rank_dict[c] for c in contestants}
        else:  # Percent - 需要假设具体票数,这里用 rank 逆序作为近似
            combined = compute_percent_combined(judge_scores, fan_rank_dict)
        
        # 检验: 淘汰者的combined应该是最高(最差)
        worst = max(combined.values())
        if combined[eliminated_id] == worst:
            # 检查是否唯一最差(避免并列情况)
            if list(combined.values()).count(worst) == 1:
                feasible.append(fan_rank_dict)
    
    return feasible
```

### 4.3 输出
对于每个选手 $i$：
$$
\hat{V}_{i,w} \in [\min_{\Pi \in \text{Feasible}} \Pi(i), \max_{\Pi \in \text{Feasible}} \Pi(i)]
$$

---

## 五、模型 M2：贝叶斯流动性推断

### 5.1 先验模型 (Prior)

基于选手属性构建粉丝投票的先验分布：

$$
\log V_{i,w} \sim \mathcal{N}(\mu_i, \sigma^2)
$$

其中：
$$
\mu_i = \beta_0 + \beta_1 \cdot \text{Industry}_i + \beta_2 \cdot \text{Age}_i + \beta_3 \cdot \text{ProPartner}_i + \beta_4 \cdot J_{i,w}
$$

### 5.2 似然函数 (Likelihood)

$$
P(\text{Eliminated}_w = e | V_1, ..., V_n) = \mathbb{1}[\arg\max R(V) = e]
$$

### 5.3 后验推断 (MCMC)

使用 **Approximate Bayesian Computation (ABC)** 或 **Gibbs Sampling**：

```
Initialize V^(0) from prior
For t = 1 to T:
    For each contestant i:
        Propose V'_i ~ q(V_i | V^(t-1)_i)
        Compute acceptance ratio α based on constraint satisfaction
        Accept/Reject with probability min(1, α)
    V^(t) = updated votes
Return posterior samples {V^(1), ..., V^(T)}
```

### 5.4 后验统计量

- **点估计**：$\hat{V}_{i,w} = \text{median}(V^{(t)}_{i,w})$
- **置信区间**：$95\% \text{CI} = [Q_{2.5\%}, Q_{97.5\%}]$
- **流动性深度**：$\text{HLD}_i = \text{median}(\text{FanRank}_i)$

---

## 六、模型 M3：选手属性影响因子分析

### 6.1 目标
回答题目要求：*"How much do such things impact how well a celebrity will do?"*

### 6.2 模型结构

**第一阶段**：推断 $\hat{V}_{i,w}$（使用 M1 或 M2）

**第二阶段**：回归分析
$$
\hat{V}_{i,w} = f(\text{Industry}, \text{Age}, \text{HomeState}, \text{ProPartner}, \text{Week}, \epsilon)
$$

使用：
- **XGBoost + SHAP**：非线性关系 + 可解释性
- **Hierarchical Linear Model**：控制季节效应

### 6.3 关键假设检验

| 假设 | 模型检验 |
|------|----------|
| Athletes get more fan votes | $\beta_{\text{Athlete}} > 0$ |
| Younger celebrities are more popular | $\beta_{\text{Age}} < 0$ |
| Top pro dancers boost fan votes | $\beta_{\text{ProPartner:Derek Hough}} > 0$ |
| Fan votes and judge scores are correlated | $\text{Corr}(\hat{V}, J) > 0$ |

---

## 七、模型 M4：投票机制公平性分析

### 7.1 目标
回答题目要求：*"If differences in outcomes exist, does one method seem to favor fan votes more than the other?"*

### 7.2 反事实分析 (Counterfactual Analysis)

对于每季，使用推断的 $\hat{V}$：
```
For each season s:
    For each week w:
        Compute Rank-Based elimination: e_rank
        Compute Percent-Based elimination: e_percent
        If e_rank != e_percent:
            Record discrepancy
            Analyze which method favored whom
```

### 7.3 杠杆效应量化

定义**粉丝杠杆指数 (Fan Leverage Index, FLI)**：
$$
\text{FLI}_M = \frac{\text{Var}(\text{FinalPlacement} | \text{FanRank})}{\text{Var}(\text{FinalPlacement} | \text{JudgeRank})}
$$

- $\text{FLI}_{\text{Rank}} > \text{FLI}_{\text{Percent}}$ → Rank 机制更偏向粉丝
- 反之亦然

---

## 八、验证与鲁棒性

### 8.1 内部一致性检验

对于已知结果（如最终冠军），检验推断的 $\hat{V}$ 是否与最终排名一致。

### 8.2 争议案例压力测试

| 争议选手 | 季 | 预期模型输出 |
|----------|---|--------------|
| Jerry Rice | 2 | HLD 极低（铁粉托底） |
| Billy Ray Cyrus | 4 | HLD 极低 |
| Bristol Palin | 11 | HLD 赛季最低 |
| Bobby Bones | 27 | HLD 极低 + 粉丝杠杆最高 |

### 8.3 敏感性分析

- 变动先验参数 $\sigma^2$ ± 20%
- 变动约束松弛度（允许近似满足）
- 对比不同 MCMC 采样链的收敛性

---

## 九、论文结构建议

```
1. Introduction
   - DWTS as an information-asymmetric voting system
   - The "hidden liquidity" metaphor from finance

2. Model Development
   - M1: Constraint-based permutation inference
   - M2: Bayesian liquidity inference with MCMC
   - M3: Factor analysis (celebrity attributes)
   - M4: Mechanism fairness comparison

3. Results
   - Fan vote estimates with confidence intervals
   - Controversy case studies (Bristol, Bobby, etc.)
   - Rank vs Percent mechanism comparison
   - Celebrity/Pro impact factors (SHAP plots)

4. Sensitivity Analysis
   - Prior robustness
   - Constraint relaxation
   - MCMC convergence diagnostics

5. Recommendations
   - Which mechanism is "fairer"?
   - Proposed hybrid mechanism
   - Memo to DWTS producers

6. Conclusion
```

---

## 十、三天实施路线图

| 天数 | 任务 | 产出 |
|------|------|------|
| **Day 1** | 数据清洗 + M1 约束求解器 | 每周可行粉丝排名集 |
| **Day 2** | M2 贝叶斯推断 + M3 因子分析 | $\hat{V}$ 点估计 + 置信区间 + SHAP |
| **Day 3** | M4 机制对比 + 争议案例 + 敏感性 | 完整论文草稿 |

---

> **Killer Quote for Abstract**:  
> *"By reconceptualizing DWTS as a dual-auction market where judges provide 'public quotes' and fans submit 'dark pool orders', we reverse-engineer the hidden voting distribution through survival-based liquidity inference—revealing that Bristol Palin commanded the deepest latent fan liquidity in the show's history, equivalent to a market-maker's undisclosed iceberg order."*
