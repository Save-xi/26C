# Hidden Liquidity Inference Framework for DWTS Fan Vote Estimation (v2.0)

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

**H1 (理性淘汰)**：每周综合得分最低者被淘汰（机制相关）。

**H2 (隐藏流动性)**：粉丝投票 $V_{i,w}$ 不可直接观测，但其效应体现在选手的"存活韧性"中。

**H3 (噪声容忍)**：允许约束松弛 $\delta \in [0, 0.5]$，以应对裁判偏差、规则扰动、平票打破等现实情况。

---

## 二、投票机制精确刻画

### 2.1 三种机制定义

| 机制 | 适用季 | 综合分公式 | 淘汰规则 |
|------|--------|-----------|----------|
| **Rank** | S1-2 | $C_i = \text{JudgeRank}_i + \text{FanRank}_i$ | $\arg\max C_i$ 被淘汰 |
| **Percent** | S3-27 | $C_i = \frac{J_i}{\sum_j J_j} + \frac{V_i}{\sum_j V_j}$ | $\arg\min C_i$ 被淘汰 |
| **Rank+JudgePick** | S28-34 | 同 Rank 取 $\text{Bottom}_2$ | 评委从 $\text{Bottom}_2$ 中二选一淘汰 |

### 2.2 Rank+JudgePick 两阶段约束

```
Stage 1: 确定 Bottom2
    Bottom2 = {a, b} s.t. C_a, C_b 为最大两个

Stage 2: 评委裁决
    eliminated ∈ Bottom2  (由评委现场投票决定)
```

**约束形式化**：
- 对所有 $k \notin \text{Bottom}_2$：$C_k < \min(C_a, C_b) - \delta$
- $\text{eliminated} \in \{a, b\}$

### 2.3 机制切换表

```python
SEASON_MODE = {
    1: "Rank", 2: "Rank",
    **{s: "Percent" for s in range(3, 28)},
    **{s: "Rank+JudgePick" for s in range(28, 35)}
}
```

---

## 三、数据处理与事件序列

### 3.1 清洗规则

| 异常类型 | 处理方法 |
|----------|----------|
| **N/A 评分** | 缺第4评委 → 用前3评委均值；整周缺失 → 标记 `mask=True` |
| **多舞平均** | 原数据已均值化，直接使用（如 8.5 表示两舞均分） |
| **Bonus 分** | 均匀分摊至各评委列（如 bonus=1.5 分给3评委，各加0.5） |
| **淘汰后 0 分** | 用于识别淘汰周次；建模时对后续周屏蔽该选手 |
| **周数不一致** | 按实际参赛周数构建事件序列，不强制对齐 |

### 3.2 异常周处理

| 事件类型 | 识别方法 | 处理策略 |
|----------|----------|----------|
| **无淘汰周** | 当周无人变0分 | 跳过该周约束 |
| **多人淘汰** | 当周多人变0分 | 联合约束：所有淘汰者均为最差 |
| **中途退赛** | results 含 "Withdrew" | 标记 `withdrawal=True`，不计入约束 |
| **全明星赛 S15** | 特殊赛季 | 单独处理或排除 |

### 3.3 事件表结构 (per season)

```python
@dataclass
class WeekEvent:
    season: int
    week: int
    contestants_alive: List[str]      # 当周存活选手
    judge_scores: Dict[str, float]    # 评委总分
    eliminated: Optional[str]         # 淘汰者 (None if no elimination)
    bottom2: Optional[Tuple[str,str]] # S28+ 底两名
    mechanism: str                    # Rank | Percent | Rank+JudgePick
    is_finale: bool                   # 是否决赛周
    special_flags: Set[str]           # {"no_elim", "double_elim", "withdrawal"}
```

---

## 四、约束构建与求解策略

### 4.1 约束系统形式化

**Rank 模式约束**：
$$
\forall i \neq e_w: \quad \text{JudgeRank}_i + \text{FanRank}_i < \text{JudgeRank}_{e_w} + \text{FanRank}_{e_w} - \delta
$$

**Percent 模式约束**：
$$
\forall i \neq e_w: \quad \frac{J_i}{\sum J} + \frac{V_i}{\sum V} > \frac{J_{e_w}}{\sum J} + \frac{V_{e_w}}{\sum V} + \delta
$$

**Rank+JudgePick 约束**：
$$
\begin{cases}
e_w \in \text{Bottom}_2 \\
\forall k \notin \text{Bottom}_2: C_k < \min(C_a, C_b) - \delta
\end{cases}
$$

### 4.2 分治求解策略

| 选手数 $n$ | 策略 | 复杂度 |
|------------|------|--------|
| $n \leq 6$ | 全排列枚举 | $O(n!)$ ≤ 720 |
| $6 < n \leq 12$ | CP-SAT 约束求解 | $O(\text{poly}(n))$ |
| $n > 12$ | MCMC/SMC 采样 | $O(T \cdot n)$ |

### 4.3 CP-SAT 形式化 (Google OR-Tools)

```python
def solve_week_cpsat(judge_scores, eliminated, mechanism, relax=0.0):
    """
    使用 CP-SAT 求解可行粉丝排名
    
    Variables:
        fan_rank[i] ∈ [1, n]  for each contestant i
    
    Constraints:
        AllDifferent(fan_rank)
        combined[eliminated] >= combined[i] + relax  for all i ≠ eliminated
    """
    from ortools.sat.python import cp_model
    
    model = cp_model.CpModel()
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    # Variables: fan_rank[i] ∈ [1, n]
    fan_rank = {c: model.NewIntVar(1, n, f'fan_{c}') for c in contestants}
    
    # AllDifferent constraint
    model.AddAllDifferent(list(fan_rank.values()))
    
    # Judge ranks (1 = best)
    sorted_by_judge = sorted(contestants, key=lambda c: -judge_scores[c])
    judge_rank = {c: i+1 for i, c in enumerate(sorted_by_judge)}
    
    # Combined score constraint
    for c in contestants:
        if c != eliminated:
            # combined[elim] > combined[c]
            # judge_rank[elim] + fan_rank[elim] > judge_rank[c] + fan_rank[c]
            model.Add(
                judge_rank[eliminated] + fan_rank[eliminated] 
                > judge_rank[c] + fan_rank[c] + int(relax)
            )
    
    # Enumerate all solutions
    solver = cp_model.CpSolver()
    solution_collector = SolutionCollector(fan_rank, contestants)
    solver.SearchForAllSolutions(model, solution_collector)
    
    return solution_collector.solutions
```

### 4.4 MCMC 采样 (大 $n$ 场景)

```python
def mcmc_sample_votes(judge_scores, eliminated, mechanism, 
                      T=5000, burn=1000, thin=5):
    """
    Metropolis-Hastings 采样粉丝排名
    
    Proposal: 交换两个选手的 fan_rank
    Acceptance: 满足约束 → accept; 否则以 exp(-λ·violation) 概率接受
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    # Initialize: fan_rank inversely proportional to judge_score
    current = initialize_from_prior(judge_scores)
    samples = []
    
    for t in range(T):
        # Propose: swap two random ranks
        i, j = random.sample(range(n), 2)
        proposal = current.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]
        
        # Check constraint satisfaction
        curr_violation = compute_violation(current, judge_scores, eliminated, mechanism)
        prop_violation = compute_violation(proposal, judge_scores, eliminated, mechanism)
        
        # Acceptance probability
        if prop_violation == 0:
            accept = True
        elif curr_violation == 0:
            accept = random.random() < math.exp(-LAMBDA * prop_violation)
        else:
            accept = random.random() < math.exp(LAMBDA * (curr_violation - prop_violation))
        
        if accept:
            current = proposal
        
        if t >= burn and (t - burn) % thin == 0:
            samples.append(current.copy())
    
    return samples
```

---

## 五、贝叶斯流动性推断

### 5.1 层次先验模型

$$
\log V_{i,w} \sim \mathcal{N}(\mu_{i,w}, \sigma^2)
$$

其中：
$$
\mu_{i,w} = \underbrace{\beta_0}_{\text{intercept}} 
+ \underbrace{\beta_{\text{ind}} \cdot \mathbf{X}_{\text{industry}}}_{\text{行业效应}}
+ \underbrace{\beta_{\text{age}} \cdot \text{Age}_i}_{\text{年龄效应}}
+ \underbrace{\beta_J \cdot J_{i,w}}_{\text{评委分关联}}
+ \underbrace{u_{\text{pro}[i]}}_{\text{舞伴随机效应}}
+ \underbrace{u_{\text{season}}}_{\text{季节随机效应}}
+ \underbrace{\gamma_w}_{\text{周效应}}
+ \underbrace{\phi \cdot \text{Momentum}_{i,w}}_{\text{动量项}}
$$

### 5.2 动态特征

**动量 (Momentum)**：
$$
\text{Momentum}_{i,w} = \alpha_1 \cdot \Delta J_{i,w-1} + \alpha_2 \cdot \Delta\text{Rank}_{i,w-1}
$$

**争议指示变量**：
$$
\text{Controversy}_{i,w} = \mathbb{1}[\text{JudgeRank}_{i,w} \geq n-1 \land S_{i,w} = 1]
$$

### 5.3 舞伴层次效应

```
pro_effect[p] ~ N(0, τ²_pro)

Top-tier pros (Derek Hough, Mark Ballas, Val, etc.):
    Expected u_pro > 0 (boost fan votes)

New/lesser-known pros:
    Expected u_pro ≈ 0
```

---

## 六、粉丝票量化与不确定性

### 6.1 输出规范

对每个选手 $i$ 每周 $w$：

| 指标 | 定义 | 用途 |
|------|------|------|
| `FanRank_median` | $\text{median}(\text{FanRank}^{(t)}_{i,w})$ | 点估计 |
| `FanRank_CI` | $[Q_{2.5\%}, Q_{97.5\%}]$ | 95% 置信区间 |
| `HLD` (Hidden Liquidity Depth) | $\mathbb{E}[\text{FanRank}_i]$ | 流动性深度 |
| `FanShare` | $\hat{V}_{i,w} / \sum_j \hat{V}_{j,w}$ | 粉丝份额（归一化） |

### 6.2 总票数标定

由于真实总票数未知，设：
$$
\sum_{i \in \text{Alive}_w} V_{i,w} = T_w \quad (\text{可调超参，默认} T_w = 1)
$$

若需绝对量级，可参考历史公开数据（如 S1 公开的百万级投票）进行校准。

---

## 七、评估与校准指标

### 7.1 淘汰预测评估

| 指标 | 公式 | 含义 |
|------|------|------|
| **淘汰复原率** | $\frac{1}{W}\sum_w \mathbb{1}[\hat{e}_w = e_w]$ | 预测淘汰者准确率 |
| **Brier Score** | $\frac{1}{nW}\sum_{i,w}(p_{i,w} - y_{i,w})^2$ | 概率校准度 |
| **Log-Loss** | $-\frac{1}{nW}\sum_{i,w}[y\log p + (1-y)\log(1-p)]$ | 似然评估 |
| **区间覆盖率** | $\frac{1}{W}\sum_w \mathbb{1}[e_w \in \text{Top-k-Risk}]$ | 真实淘汰落入高风险集比例 |

### 7.2 置信区间评估

```python
def evaluate_ci_coverage(samples, true_outcomes):
    """
    检验 95% CI 的真实覆盖率
    理想情况下应接近 95%
    """
    covered = 0
    for week, true_rank in true_outcomes.items():
        ci_low, ci_high = np.percentile(samples[week], [2.5, 97.5])
        if ci_low <= true_rank <= ci_high:
            covered += 1
    return covered / len(true_outcomes)
```

---

## 八、公平性与机制对比

### 8.1 反事实重放算法

```python
def counterfactual_analysis(season_data, posterior_samples):
    """
    对每季每周，使用推断的 V 在不同机制下重放淘汰决策
    """
    results = []
    
    for week_event in season_data:
        for V_sample in posterior_samples[week_event]:
            # 计算三种机制下的淘汰者
            elim_rank = compute_elimination(V_sample, "Rank")
            elim_percent = compute_elimination(V_sample, "Percent")
            elim_jp = compute_elimination(V_sample, "Rank+JudgePick")
            
            results.append({
                'week': week_event.week,
                'actual': week_event.eliminated,
                'rank_cf': elim_rank,
                'percent_cf': elim_percent,
                'jp_cf': elim_jp,
                'discrepancy': len(set([elim_rank, elim_percent, elim_jp])) > 1
            })
    
    return pd.DataFrame(results)
```

### 8.2 杠杆/弹性指标

**评委弹性 (Judge Elasticity)**：
$$
\eta_J = \frac{\partial \Pr(\text{survive})}{\partial J} \bigg|_{\bar{J}, \bar{V}}
$$

**粉丝弹性 (Fan Elasticity)**：
$$
\eta_V = \frac{\partial \Pr(\text{survive})}{\partial V} \bigg|_{\bar{J}, \bar{V}}
$$

**机制偏向指数**：
$$
\text{Bias}_M = \frac{\eta_V}{\eta_J} \quad
\begin{cases}
> 1 & \text{偏粉丝} \\
< 1 & \text{偏评委} \\
= 1 & \text{平衡}
\end{cases}
$$

### 8.3 争议案例量化表

| Season | Contestant | JudgeRank (周均) | HLD Percentile | 机制分歧 | 反事实结果 |
|--------|------------|------------------|----------------|----------|------------|
| 2 | Jerry Rice | 倒数1-2 | P5 | Rank ≠ Percent | Percent下更早淘汰 |
| 4 | Billy Ray Cyrus | 倒数1-2 | P10 | 有 | - |
| 11 | Bristol Palin | 倒数 (12次) | **P1** (最强) | 显著 | - |
| 27 | Bobby Bones | 倒数 | P3 | N/A (Rank+JP) | - |

---

## 九、可视化产出

### 9.1 推荐图表

| 图表类型 | 内容 | 用途 |
|----------|------|------|
| **瀑布图** | Combined 分解为 Judge vs Fan 贡献 | 解释每周淘汰原因 |
| **生存曲线** | 基于 posterior 淘汰概率的 Kaplan-Meier | 选手"寿命"分布 |
| **误差条图** | FanRank 点估计 + 95% CI | 不确定性展示 |
| **SHAP Summary** | 特征对 FanRank 的影响 | 因子重要性 |
| **热力图** | 季×选手 的 HLD 矩阵 | 跨季对比 |
| **分歧率曲线** | Rank vs Percent 机制分歧随周数变化 | 机制对比 |

### 9.2 可视化代码提示

```python
# 瀑布图示例
def plot_combined_waterfall(week_data, fan_rank_estimate):
    """
    展示单周各选手的 combined score 分解
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    contestants = week_data.contestants_alive
    judge_contrib = [compute_judge_contrib(c, week_data) for c in contestants]
    fan_contrib = [compute_fan_contrib(c, fan_rank_estimate) for c in contestants]
    
    x = np.arange(len(contestants))
    ax.bar(x, judge_contrib, label='Judge Contribution', color='steelblue')
    ax.bar(x, fan_contrib, bottom=judge_contrib, label='Fan Contribution', color='coral')
    
    ax.axhline(y=threshold, linestyle='--', color='red', label='Elimination Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(contestants, rotation=45)
    ax.legend()
    
    return fig
```

---

## 十、代码接口规范

### 10.1 核心求解器接口

```python
def solve_week(
    judge_scores: Dict[str, float],
    eliminated: str,
    mechanism: str,           # "Rank" | "Percent" | "Rank+JudgePick"
    bottom2: Optional[Tuple[str, str]] = None,  # for Rank+JudgePick
    relax: float = 0.0,       # constraint relaxation δ
    solver: str = "auto"      # "enum" | "cpsat" | "mcmc"
) -> List[Dict[str, int]]:
    """
    Returns: List of feasible fan_rank assignments
    """
    n = len(judge_scores)
    
    if solver == "auto":
        solver = "enum" if n <= 6 else ("cpsat" if n <= 12 else "mcmc")
    
    if solver == "enum":
        return solve_week_enum(judge_scores, eliminated, mechanism, relax)
    elif solver == "cpsat":
        return solve_week_cpsat(judge_scores, eliminated, mechanism, relax)
    else:
        return mcmc_sample_votes(judge_scores, eliminated, mechanism)
```

### 10.2 采样接口

```python
def sample_posterior(
    season_events: List[WeekEvent],
    prior_params: Dict,
    T: int = 5000,
    burn: int = 1000,
    thin: int = 5
) -> Dict[int, np.ndarray]:
    """
    Returns: {week: (n_samples, n_contestants) array of fan_rank samples}
    """
    pass
```

### 10.3 评估接口

```python
def evaluate_model(
    predictions: Dict[int, np.ndarray],
    ground_truth: Dict[int, str]
) -> Dict[str, float]:
    """
    Returns: {
        'accuracy': float,
        'brier_score': float,
        'log_loss': float,
        'ci_coverage': float
    }
    """
    pass
```

---

## 十一、三天实施路线图（更新版）

| 天数 | 上午 | 下午 | 产出 |
|------|------|------|------|
| **Day 1** | 数据清洗 + 事件序列构建 | M1 约束求解器 (enum + cpsat) | 每周可行粉丝排名集 |
| **Day 2** | M2 MCMC 采样器 + 先验标定 | M3 因子分析 (层次模型) | 后验分布 + SHAP |
| **Day 3** | M4 机制对比 + 争议案例 | 敏感性分析 + 论文撰写 | 完整论文草稿 |

---

## 十二、关键技术决策备忘

| 决策点 | 推荐选项 | 备选方案 | 理由 |
|--------|----------|----------|------|
| 约束求解 | CP-SAT | ILP (Gurobi) | 开源 + 足够快 |
| MCMC 采样 | Metropolis-Hastings | SMC / NUTS | 实现简单 |
| 先验 | 层次正态 | Dirichlet | 可解释性强 |
| 敏感性 | δ ∈ {0, 0.1, 0.3, 0.5} | - | 覆盖从严到松 |
| 评估 | Brier + 复原率 | AUC | 直观 + 可比 |

---

> **Killer Quote for Abstract (v2)**:  
> *"By reconceptualizing DWTS as a dual-auction market with observable 'public quotes' (judge scores) and latent 'dark pool orders' (fan votes), we employ constraint-satisfaction programming and Bayesian inference to reverse-engineer the hidden voting distribution. Our model reveals Bristol Palin commanded the deepest latent fan liquidity in the show's 34-season history—surviving 12 elimination-worthy judge scores through an 'iceberg order' of loyal voters invisible to all but the survival outcome."*
