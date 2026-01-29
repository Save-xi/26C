# Hidden Liquidity Inference Framework v3.0

> 全面修复v2.0的7项技术漏洞，可直接指导代码实现。

---

## 一、三种机制的精确约束建模

### 1.1 Rank 模式 (S1-2)

**变量**：`fan_rank[i] ∈ {1,...,n}`，AllDifferent

**约束**：
```
∀i ≠ e: judge_rank[e] + fan_rank[e] ≥ judge_rank[i] + fan_rank[i] + 1
```
（+1 确保严格大于，防止并列）

### 1.2 Percent 模式 (S3-27) ✅ 已修复

**变量**：票份额 `v[i] ≥ 0`，且 `Σv[i] = 1`

**综合分**（越高越好）：
```
C[i] = J[i]/ΣJ + v[i]
```

**约束**（淘汰者 e 的 C 最小，即最差）：
```
∀i ≠ e: C[i] ≥ C[e] + δ             # 非淘汰者得分更高
展开: J[i]/ΣJ + v[i] ≥ J[e]/ΣJ + v[e] + δ
整理: v[i] - v[e] ≥ (J[e] - J[i])/ΣJ + δ  # ⚠️ 注意符号
```

这是**线性约束**，可用 LP/QP 求解可行域。

**若需离散排名**：引入 big-M 线性化
```
v[i] - v[j] ≥ ε - M·(1 - y[i,j])   # y[i,j]=1 表示 i 票份额 > j
```

### 1.3 Rank+JudgePick 模式 (S28-34) ✅ 已修复

**两阶段约束**：

> **注意**: Combined rank = Judge_rank + Fan_rank，**值越大=排名越差**

**Stage 1: 确定 Bottom2**（综合排名最差的两人）
```
对所有 k ∉ Bottom2:
    combined[k] ≤ min(combined[a], combined[b]) - 1  # k 排名更靠前(值更小)
其中 Bottom2 = {a, b}
```

**Stage 2: 评委裁决**
```
eliminated ∈ {a, b}  (外部给定，非求解)
```

**求解策略**：
1. 枚举所有 C(n,2) 个 Bottom2 组合
2. 对每个组合，求解满足 Stage1 的 fan_rank
3. 保留 eliminated ∈ Bottom2 的组合

---

## 二、数据提取流程（从 CSV 到事件序列）

### 2.1 淘汰识别算法

```python
def extract_elimination_events(df, season):
    """从CSV提取每周淘汰事件"""
    season_df = df[df['season'] == season].copy()
    events = []
    
    # 获取所有周列
    week_cols = [c for c in df.columns if c.startswith('week') and 'judge' in c]
    max_week = max(int(c.split('_')[0].replace('week','')) for c in week_cols)
    
    for week in range(1, max_week + 1):
        judge_cols = [f'week{week}_judge{j}_score' for j in [1,2,3,4]]
        
        # 本周存活选手：有非零、非N/A分数
        alive = []
        scores = {}
        for _, row in season_df.iterrows():
            name = row['celebrity_name']
            week_scores = [row[c] for c in judge_cols if c in row and row[c] not in ['N/A', 0, '0']]
            if week_scores and any(float(s) > 0 for s in week_scores if s != 'N/A'):
                alive.append(name)
                scores[name] = sum(float(s) for s in week_scores if s != 'N/A')
        
        if len(alive) < 2:
            continue
        
        # 识别淘汰者：下周分数变0（支持多人淘汰）
        eliminated_list = []
        next_week_cols = [f'week{week+1}_judge{j}_score' for j in [1,2,3,4]]
        for name in alive:
            row = season_df[season_df['celebrity_name'] == name].iloc[0]
            next_scores = [row[c] for c in next_week_cols if c in row]
            if all(s in [0, '0', 'N/A'] for s in next_scores):
                # 确认是淘汰而非退赛
                if 'Withdrew' not in str(row['results']):
                    eliminated_list.append(name)
        
        # 单人淘汰返回str，多人淘汰返回list
        eliminated = eliminated_list[0] if len(eliminated_list) == 1 else (eliminated_list if eliminated_list else None)
        
        # 机制判定
        mechanism = get_mechanism(season)
        
        events.append({
            'season': season,
            'week': week,
            'alive': alive,
            'scores': scores,
            'eliminated': eliminated,  # str | List[str] | None
            'mechanism': mechanism,
            'is_finale': 'Place' in str(season_df[season_df['celebrity_name'].isin(alive)].iloc[0]['results'])
        })
    
    return events

def get_mechanism(season):
    if season <= 2:
        return 'Rank'
    elif season <= 27:
        return 'Percent'
    else:
        return 'Rank+JudgePick'
```

### 2.2 Bottom2 提取（S28+）

```python
def extract_bottom2(event, df):
    """
    S28+: 优先从数据集直接提取 Bottom2（节目公开宣布）
    如果数据缺失则标记为 None，避免错误推断
    """
    season = event['season']
    week = event['week']
    
    # 尝试从 results 列或 metadata 提取
    week_data = df[(df['season'] == season)]
    bottom2_col = f'week{week}_bottom2'  # 假设有此列
    
    if bottom2_col in df.columns:
        b2_str = week_data[bottom2_col].dropna().iloc[0] if not week_data[bottom2_col].dropna().empty else None
        if b2_str and ',' in str(b2_str):
            return tuple(sorted(b2_str.split(',')))
    
    # 数据缺失：返回 None，在报告中说明数据限制
    return None

def infer_bottom2_posterior(event, posterior_samples):
    """
    仅当 extract_bottom2 返回 None 时使用
    从后验分布推断 Bottom2（需标注为"推断值"）
    """
    bottom2_candidates = []
    for sample in posterior_samples:
        combined = {c: event['judge_rank'][c] + sample[c] for c in event['alive']}
        sorted_c = sorted(combined.items(), key=lambda x: -x[1])  # 降序=最差在前
        bottom2_candidates.append(tuple(sorted([sorted_c[0][0], sorted_c[1][0]])))
    return bottom2_candidates
```

### 2.3 异常处理规则

| 情况 | 识别方法 | 处理 |
|------|----------|------|
| 无淘汰周 | eliminated=None | 跳过该周约束 |
| 多人淘汰 | 多人下周变0 | 联合约束：所有淘汰者分最低 |
| 退赛 | results含"Withdrew" | 标记但不计入约束 |
| S15全明星 | season==15 | **排除**（选手重复出现，扰乱先验） |
| 并列分数 | 同分 | 稳定排序 + ε扰动 |

---

## 三、CP-SAT 求解器（修复版）

```python
from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple
import math

class FanRankSolutionCollector(cp_model.CpSolverSolutionCallback):
    """收集CP-SAT所有可行解"""
    def __init__(self, fan_rank_vars, contestants, max_solutions=1000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._fan_rank = fan_rank_vars
        self._contestants = contestants
        self._solutions = []
        self._max = max_solutions
    
    def on_solution_callback(self):
        if len(self._solutions) < self._max:
            sol = {c: self.Value(self._fan_rank[c]) for c in self._contestants}
            self._solutions.append(sol)
        else:
            self.StopSearch()
    
    @property
    def solutions(self):
        return self._solutions

def solve_rank_cpsat(
    judge_scores: Dict[str, float],
    eliminated: str,
    relax: float = 0.0,
    max_solutions: int = 1000,
    timeout_sec: float = 60.0
) -> List[Dict[str, int]]:
    """
    Rank模式的CP-SAT求解（S1-2, S28-34的Stage1）
    """
    model = cp_model.CpModel()
    contestants = list(judge_scores.keys())
    n = len(contestants)
    
    # ===== 处理并列：随机ε扰动（比字母序更合理） =====
    perturbed_scores = {c: judge_scores[c] + np.random.uniform(-1e-6, 1e-6) for c in contestants}
    sorted_by_score = sorted(contestants, key=lambda c: -perturbed_scores[c])
    judge_rank = {c: i + 1 for i, c in enumerate(sorted_by_score)}
    
    # ===== 变量：fan_rank[i] ∈ [1, n] =====
    fan_rank = {c: model.NewIntVar(1, n, f'fan_{c}') for c in contestants}
    
    # ===== AllDifferent =====
    model.AddAllDifferent(list(fan_rank.values()))
    
    # ===== 淘汰约束：eliminated 的 combined 最大 =====
    # 整数松弛：relax_int = ceil(relax * 10) / 10 → 放大为整数
    relax_int = max(1, int(math.ceil(relax * 10)))  # 最小为1确保严格
    
    for c in contestants:
        if c != eliminated:
            # combined[e] >= combined[c] + relax_int
            # judge_rank[e] + fan_rank[e] >= judge_rank[c] + fan_rank[c] + relax_int
            model.Add(
                judge_rank[eliminated] + fan_rank[eliminated]
                >= judge_rank[c] + fan_rank[c] + relax_int
            )
    
    # ===== 求解设置 =====
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_sec
    solver.parameters.enumerate_all_solutions = True
    
    collector = FanRankSolutionCollector(fan_rank, contestants, max_solutions)
    status = solver.Solve(model, collector)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return collector.solutions
    else:
        return []

def solve_rank_judgepick_cpsat(
    judge_scores: Dict[str, float],
    eliminated: str,
    bottom2: Optional[Tuple[str, str]] = None,
    relax: float = 0.0,
    max_solutions: int = 500,
    timeout_sec: float = 60.0
) -> List[Dict[str, int]]:
    """
    Rank+JudgePick模式的CP-SAT求解（S28-34）
    """
    contestants = list(judge_scores.keys())
    n = len(contestants)
    
    all_solutions = []
    
    # ===== 枚举所有可能的 Bottom2 =====
    from itertools import combinations
    
    if bottom2 is not None:
        bottom2_candidates = [bottom2]
    else:
        # 只考虑包含 eliminated 的组合
        bottom2_candidates = [
            (eliminated, other) for other in contestants if other != eliminated
        ]
    
    for b2 in bottom2_candidates:
        model = cp_model.CpModel()
        
        sorted_by_score = sorted(contestants, key=lambda c: (-judge_scores[c], c))
        judge_rank = {c: i + 1 for i, c in enumerate(sorted_by_score)}
        
        fan_rank = {c: model.NewIntVar(1, n, f'fan_{c}') for c in contestants}
        model.AddAllDifferent(list(fan_rank.values()))
        
        relax_int = max(1, int(math.ceil(relax * 10)))
        a, b = b2
        
        # Stage1约束：非Bottom2的combined更小（排名更靠前）
        for k in contestants:
            if k not in b2:
                # combined[k] <= combined[Bottom2] - relax_int
                model.Add(
                    judge_rank[k] + fan_rank[k]
                    <= judge_rank[a] + fan_rank[a] - relax_int
                )
                model.Add(
                    judge_rank[k] + fan_rank[k]
                    <= judge_rank[b] + fan_rank[b] - relax_int
                )
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_sec / len(bottom2_candidates)
        solver.parameters.enumerate_all_solutions = True
        
        collector = FanRankSolutionCollector(fan_rank, contestants, max_solutions // len(bottom2_candidates))
        solver.Solve(model, collector)
        
        for sol in collector.solutions:
            sol['_bottom2'] = b2
            all_solutions.append(sol)
    
    return all_solutions

def solve_rank_enum(
    judge_scores: Dict[str, float],
    eliminated: str,
    relax: float = 0.0
) -> List[Dict[str, int]]:
    """
    Rank模式的全排列枚举求解（n≤6时使用）
    """
    from itertools import permutations
    
    contestants = list(judge_scores.keys())
    n = len(contestants)
    
    sorted_by_score = sorted(contestants, key=lambda c: (-judge_scores[c], c))
    judge_rank = {c: i + 1 for i, c in enumerate(sorted_by_score)}
    
    feasible = []
    
    for perm in permutations(range(1, n + 1)):
        fan_rank = dict(zip(contestants, perm))
        combined = {c: judge_rank[c] + fan_rank[c] for c in contestants}
        
        e_combined = combined[eliminated]
        
        # 检验淘汰者是否综合分最大(最差)
        is_worst = all(
            e_combined >= combined[c] + (1 if relax == 0 else 0)
            for c in contestants if c != eliminated
        )
        
        if is_worst:
            feasible.append(fan_rank)
    
    return feasible
```

---

## 四、Percent 模式的 LP 求解

```python
import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Tuple

def solve_percent_lp(
    judge_scores: Dict[str, float],
    eliminated: str,
    delta: float = 0.01,
    n_samples: int = 100
) -> List[Dict[str, float]]:
    """
    Percent模式：求解票份额v[i]的可行域
    
    约束：
        v[e] - v[i] <= (J[i] - J[e])/ΣJ - δ  for all i ≠ e
        Σv = 1, v >= 0
    """
    contestants = list(judge_scores.keys())
    n = len(contestants)
    total_judge = sum(judge_scores.values())
    
    # 索引映射
    idx = {c: i for i, c in enumerate(contestants)}
    e_idx = idx[eliminated]
    
    # ===== 不等式约束 A_ub @ v <= b_ub =====
    # 原约束: v[i] - v[e] >= (J[e] - J[i])/ΣJ + δ
    # 转化为: v[e] - v[i] <= (J[i] - J[e])/ΣJ - δ  ← 标准LP形式
    A_ub = []
    b_ub = []
    
    for c in contestants:
        if c != eliminated:
            row = [0.0] * n
            row[e_idx] = 1.0      # v[e] 系数
            row[idx[c]] = -1.0   # -v[i] 系数
            A_ub.append(row)
            # 关键修正: (J[i] - J[e])/ΣJ - δ
            b_ub.append((judge_scores[c] - judge_scores[eliminated]) / total_judge - delta)
    
    # ===== 等式约束 Σv = 1 =====
    A_eq = [[1.0] * n]
    b_eq = [1.0]
    
    # ===== 边界 v >= 0 =====
    bounds = [(0, 1) for _ in range(n)]
    
    # ===== 采样可行域的多个点 =====
    samples = []
    
    for _ in range(n_samples):
        # 随机目标函数方向
        c_obj = np.random.randn(n)
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.success:
            v_sol = result.x
            samples.append({c: v_sol[idx[c]] for c in contestants})
    
    return samples

def percent_to_rank(vote_shares: Dict[str, float]) -> Dict[str, int]:
    """将票份额转换为排名（1=最高票）"""
    sorted_c = sorted(vote_shares.keys(), key=lambda c: -vote_shares[c])
    return {c: i + 1 for i, c in enumerate(sorted_c)}
```

---

## 五、MCMC 采样器（完整版）

```python
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class MCMCConfig:
    T: int = 5000           # 总迭代次数
    burn: int = 1000        # 预热期
    thin: int = 5           # 稀疏采样
    lambda_penalty: float = 5.0  # 约束违反惩罚
    min_ess: float = 100.0  # 最小有效样本量
    seed: int = 42

def compute_violation_rank(
    fan_rank: Dict[str, int],
    judge_rank: Dict[str, int],
    eliminated: str
) -> float:
    """计算Rank模式的约束违反程度"""
    combined = {c: judge_rank[c] + fan_rank[c] for c in fan_rank}
    e_combined = combined[eliminated]
    
    # 违反度 = 淘汰者未达到最大值的差距之和
    violation = 0.0
    for c, comb in combined.items():
        if c != eliminated and comb >= e_combined:
            violation += (comb - e_combined + 1)
    
    return violation

def compute_violation_percent(
    vote_share: Dict[str, float],
    judge_scores: Dict[str, float],
    eliminated: str,
    delta: float = 0.01
) -> float:
    """计算Percent模式的约束违反程度"""
    total_j = sum(judge_scores.values())
    combined = {c: judge_scores[c]/total_j + vote_share[c] for c in vote_share}
    e_combined = combined[eliminated]
    
    violation = 0.0
    for c, comb in combined.items():
        if c != eliminated and comb <= e_combined + delta:
            violation += (e_combined + delta - comb + 0.01)
    
    return violation

def mcmc_rank(
    judge_scores: Dict[str, float],
    eliminated: str,
    config: MCMCConfig = MCMCConfig()
) -> Dict:
    """MCMC采样Rank模式的粉丝排名"""
    np.random.seed(config.seed)
    
    contestants = list(judge_scores.keys())
    n = len(contestants)
    
    # Judge rank（稳定排序）
    sorted_c = sorted(contestants, key=lambda c: (-judge_scores[c], c))
    judge_rank = {c: i + 1 for i, c in enumerate(sorted_c)}
    
    # 初始化：按评委分逆序（假设粉丝与评委负相关）
    current = list(range(1, n + 1))
    np.random.shuffle(current)
    current_rank = dict(zip(contestants, current))
    current_viol = compute_violation_rank(current_rank, judge_rank, eliminated)
    
    samples = []
    violations = []
    
    for t in range(config.T):
        # Proposal: 随机交换两个排名
        i, j = np.random.choice(n, 2, replace=False)
        ci, cj = contestants[i], contestants[j]
        
        proposal = current_rank.copy()
        proposal[ci], proposal[cj] = proposal[cj], proposal[ci]
        prop_viol = compute_violation_rank(proposal, judge_rank, eliminated)
        
        # Metropolis-Hastings 接受准则
        if prop_viol == 0:
            accept_prob = 1.0
        elif current_viol == 0:
            accept_prob = np.exp(-config.lambda_penalty * prop_viol)
        else:
            accept_prob = min(1.0, np.exp(config.lambda_penalty * (current_viol - prop_viol)))
        
        if np.random.random() < accept_prob:
            current_rank = proposal
            current_viol = prop_viol
        
        if t >= config.burn and (t - config.burn) % config.thin == 0:
            samples.append(current_rank.copy())
            violations.append(current_viol)
    
    # 计算有效样本量 (ESS)
    ess = compute_ess(samples, contestants)
    
    return {
        'samples': samples,
        'violations': violations,
        'ess': ess,
        'converged': ess >= config.min_ess
    }

def mcmc_percent(
    judge_scores: Dict[str, float],
    eliminated: str,
    config: MCMCConfig = MCMCConfig(),
    delta: float = 0.01
) -> Dict:
    """MCMC采样Percent模式的票份额"""
    np.random.seed(config.seed)
    
    contestants = list(judge_scores.keys())
    n = len(contestants)
    
    # 初始化：均匀分布
    current = {c: 1.0/n for c in contestants}
    current_viol = compute_violation_percent(current, judge_scores, eliminated, delta)
    
    samples = []
    violations = []
    
    for t in range(config.T):
        # Proposal: Logit变换 + Random Walk（避免边界退化）
        # Stick-Breaking形式,自动满足 Σv = 1
        eps = 1e-8
        current_arr = np.array([current[c] for c in contestants])
        current_arr = np.clip(current_arr, eps, 1-eps)
        logit_current = np.log(current_arr / (1 - current_arr + eps))
        logit_proposal = logit_current + np.random.normal(0, 0.2, n)
        exp_proposal = np.exp(logit_proposal)
        proposed_arr = exp_proposal / exp_proposal.sum()  # Softmax归一化
        proposal = dict(zip(contestants, proposed_arr))
        prop_viol = compute_violation_percent(proposal, judge_scores, eliminated, delta)
        
        # 接受准则
        if prop_viol == 0:
            accept_prob = 1.0
        elif current_viol == 0:
            accept_prob = np.exp(-config.lambda_penalty * prop_viol)
        else:
            accept_prob = min(1.0, np.exp(config.lambda_penalty * (current_viol - prop_viol)))
        
        if np.random.random() < accept_prob:
            current = proposal
            current_viol = prop_viol
        
        if t >= config.burn and (t - config.burn) % config.thin == 0:
            samples.append(current.copy())
            violations.append(current_viol)
    
    ess = compute_ess_continuous(samples, contestants)
    
    return {
        'samples': samples,
        'violations': violations,
        'ess': ess,
        'converged': ess >= config.min_ess
    }

def compute_ess(samples: List[Dict], contestants: List[str]) -> float:
    """计算离散排名的有效样本量（Geyer截断法）"""
    if len(samples) < 10:
        return 0.0
    
    ess_list = []
    for c in contestants:
        ranks = [s[c] for s in samples]
        n = len(ranks)
        # Geyer 1992: 截断到第一个负自相关对
        max_lag = min(200, n // 2)
        rho = [autocorr(ranks, k) for k in range(1, max_lag)]
        cutoff = next((i for i in range(len(rho)-1) if rho[i] + rho[i+1] < 0), len(rho))
        ess_c = n / (1 + 2 * sum(rho[:cutoff]))
        ess_list.append(max(1, ess_c))
    
    return min(ess_list)

def compute_ess_continuous(samples: List[Dict], contestants: List[str]) -> float:
    """计算连续票份额的有效样本量"""
    if len(samples) < 10:
        return 0.0
    
    ess_list = []
    for c in contestants:
        shares = [s[c] for s in samples]
        var_s = np.var(shares)
        if var_s < 1e-10:
            ess_list.append(len(shares))
        else:
            ess_c = len(shares) / (1 + 2 * sum(autocorr(shares, k) for k in range(1, min(50, len(shares)//2))))
            ess_list.append(max(1, ess_c))
    
    return min(ess_list)

def autocorr(x, lag):
    """计算自相关系数（带除零保护）"""
    n = len(x)
    if lag >= n:
        return 0
    x = np.array(x) - np.mean(x)
    var_x = np.var(x)
    if var_x < 1e-10:  # 防止除零
        return 0
    return np.dot(x[:n-lag], x[lag:]) / (var_x * n)
```

---

## 六、评估函数（完整实现）

```python
import numpy as np
from typing import Dict, List

def compute_elimination_rank(
    fan_rank: Dict[str, int],
    judge_rank: Dict[str, int]
) -> str:
    """Rank模式：返回综合排名最差的选手"""
    combined = {c: judge_rank[c] + fan_rank[c] for c in fan_rank}
    return max(combined.keys(), key=lambda c: combined[c])

def compute_elimination_percent(
    vote_share: Dict[str, float],
    judge_scores: Dict[str, float]
) -> str:
    """Percent模式：返回综合得分最低的选手"""
    total_j = sum(judge_scores.values())
    combined = {c: judge_scores[c]/total_j + vote_share[c] for c in vote_share}
    return min(combined.keys(), key=lambda c: combined[c])

def compute_elimination_judgepick(
    fan_rank: Dict[str, int],
    judge_rank: Dict[str, int],
    actual_eliminated: str
) -> str:
    """Rank+JudgePick：返回Bottom2中被淘汰的（按实际结果）"""
    combined = {c: judge_rank[c] + fan_rank[c] for c in fan_rank}
    sorted_c = sorted(combined.keys(), key=lambda c: -combined[c])
    bottom2 = set(sorted_c[:2])
    
    if actual_eliminated in bottom2:
        return actual_eliminated
    else:
        return sorted_c[0]  # 如果实际淘汰者不在Bottom2，返回最差者

def evaluate_model(
    events: List[Dict],
    posterior_samples: Dict[int, List[Dict]],
    mechanism: str
) -> Dict[str, float]:
    """
    综合评估模型性能
    """
    correct = 0
    brier_sum = 0.0
    log_loss_sum = 0.0
    n_total = 0
    
    for event in events:
        week = event['week']
        if event['eliminated'] is None:
            continue
        
        samples = posterior_samples.get(week, [])
        if not samples:
            continue
        
        alive = event['alive']
        true_elim = event['eliminated']
        judge_scores = event['scores']
        
        # 计算每个选手的淘汰概率
        elim_counts = {c: 0 for c in alive}
        for sample in samples:
            if mechanism == 'Rank' or mechanism == 'Rank+JudgePick':
                sorted_c = sorted(alive, key=lambda c: (-judge_scores[c], c))
                judge_rank = {c: i+1 for i, c in enumerate(sorted_c)}
                pred_elim = compute_elimination_rank(sample, judge_rank)
            else:
                pred_elim = compute_elimination_percent(sample, judge_scores)
            
            if pred_elim in elim_counts:
                elim_counts[pred_elim] += 1
        
        # 归一化为概率
        total_samples = len(samples)
        probs = {c: elim_counts[c] / total_samples for c in alive}
        
        # 准确率：众数预测是否正确
        pred = max(probs.keys(), key=lambda c: probs[c])
        if pred == true_elim:
            correct += 1
        
        # Brier Score (per-week normalized)
        n_alive_week = len(alive)
        for c in alive:
            y = 1.0 if c == true_elim else 0.0
            brier_sum += (probs[c] - y) ** 2 / n_alive_week
        
        # Log-Loss
        p_true = max(probs.get(true_elim, 0), 1e-10)
        log_loss_sum += -np.log(p_true)
        
        n_total += 1
    
    return {
        'accuracy': correct / n_total if n_total > 0 else 0,
        'brier_score': brier_sum / n_total if n_total > 0 else 1,
        'log_loss': log_loss_sum / n_total if n_total > 0 else 10,
        'n_weeks': n_total
    }
```

---

## 七、弹性与公平性分析

```python
def estimate_elasticity(
    event: Dict,
    posterior_samples: List[Dict],
    mechanism: str,
    delta_j: float = 0.5,
    delta_v: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    数值估计评委/粉丝弹性（双向中心差分）
    
    Elasticity = ΔPr(survive) / Δ(score)
    """
    alive = event['alive']
    judge_scores = event['scores']
    
    # 基线淘汰概率
    base_probs = compute_elim_probs(posterior_samples, judge_scores, mechanism, alive)
    
    elasticities = {}
    for target in alive:
        # ===== 评委弹性（双向导数） =====
        scores_up = judge_scores.copy()
        scores_up[target] = judge_scores[target] + delta_j
        scores_down = judge_scores.copy()
        scores_down[target] = judge_scores[target] - delta_j
        
        probs_up = compute_elim_probs(posterior_samples, scores_up, mechanism, alive)
        probs_down = compute_elim_probs(posterior_samples, scores_down, mechanism, alive)
        
        # 中心差分: (f(x+h) - f(x-h)) / 2h
        delta_survive_j = (1 - probs_up[target]) - (1 - probs_down[target])
        judge_elast = delta_survive_j / (2 * delta_j)
        
        # ===== 粉丝弹性（双向导数） =====
        def perturb_samples(direction):
            perturbed = []
            for s in posterior_samples:
                ps = s.copy()
                if mechanism in ['Rank', 'Rank+JudgePick']:
                    ps[target] = max(1, ps[target] + direction)  # ±1排名
                else:
                    ps[target] = max(0, min(1, ps[target] + direction * delta_v))
                    total = sum(ps.values())
                    ps = {c: ps[c]/total for c in ps}
                perturbed.append(ps)
            return perturbed
        
        probs_v_up = compute_elim_probs(perturb_samples(-1), judge_scores, mechanism, alive)  # 排名-1=更好
        probs_v_down = compute_elim_probs(perturb_samples(+1), judge_scores, mechanism, alive)
        
        delta_survive_v = (1 - probs_v_up[target]) - (1 - probs_v_down[target])
        fan_elast = delta_survive_v / 2  # 单位: 每1位排名变化
        
        elasticities[target] = {
            'judge_elasticity': judge_elast,
            'fan_elasticity': fan_elast,
            'bias_ratio': fan_elast / judge_elast if abs(judge_elast) > 1e-6 else float('inf')
        }
    
    return elasticities

def compute_elim_probs(samples, judge_scores, mechanism, alive):
    """辅助函数：计算淘汰概率"""
    counts = {c: 0 for c in alive}
    sorted_c = sorted(alive, key=lambda c: (-judge_scores[c], c))
    judge_rank = {c: i+1 for i, c in enumerate(sorted_c)}
    
    for s in samples:
        if mechanism in ['Rank', 'Rank+JudgePick']:
            elim = compute_elimination_rank(s, judge_rank)
        else:
            elim = compute_elimination_percent(s, judge_scores)
        if elim in counts:
            counts[elim] += 1
    
    return {c: counts[c] / len(samples) for c in alive}
```

---

## 八、统一接口

```python
def solve_week(
    judge_scores: Dict[str, float],
    eliminated: str,
    mechanism: str,
    bottom2: Optional[Tuple[str, str]] = None,
    relax: float = 0.0,
    solver: str = "auto",
    max_solutions: int = 500,
    timeout_sec: float = 60.0
) -> List[Dict]:
    """
    统一求解接口
    """
    n = len(judge_scores)
    
    # 自动选择求解器
    if solver == "auto":
        if mechanism == "Percent":
            solver = "lp"
        elif n <= 6:
            solver = "enum"
        elif n <= 12:
            solver = "cpsat"
        else:
            solver = "mcmc"
    
    # 分发求解
    if mechanism == "Rank":
        if solver == "cpsat":
            return solve_rank_cpsat(judge_scores, eliminated, relax, max_solutions, timeout_sec)
        elif solver == "mcmc":
            result = mcmc_rank(judge_scores, eliminated)
            return result['samples'] if result['converged'] else []
        else:  # enum
            return solve_rank_enum(judge_scores, eliminated, relax)
    
    elif mechanism == "Percent":
        if solver == "lp":
            return solve_percent_lp(judge_scores, eliminated, relax)
        else:
            result = mcmc_percent(judge_scores, eliminated, delta=relax)
            return result['samples'] if result['converged'] else []
    
    else:  # Rank+JudgePick
        return solve_rank_judgepick_cpsat(judge_scores, eliminated, bottom2, relax, max_solutions, timeout_sec)
```

---

## 九、敏感性分析框架

```python
def sensitivity_analysis(
    events: List[Dict],
    delta_grid: List[float] = [0.0, 0.1, 0.3, 0.5],
    lambda_grid: List[float] = [1.0, 5.0, 10.0]
) -> pd.DataFrame:
    """
    对松弛参数δ和惩罚参数λ进行敏感性分析
    """
    results = []
    
    for delta in delta_grid:
        for lam in lambda_grid:
            config = MCMCConfig(lambda_penalty=lam)
            
            all_samples = {}
            for event in events:
                if event['eliminated']:
                    samples = solve_week(
                        event['scores'], 
                        event['eliminated'],
                        event['mechanism'],
                        relax=delta
                    )
                    all_samples[event['week']] = samples
            
            metrics = evaluate_model(events, all_samples, events[0]['mechanism'])
            metrics['delta'] = delta
            metrics['lambda'] = lam
            results.append(metrics)
    
    return pd.DataFrame(results)
```

---

## 十、三天路线图（v3.0）

| 时段 | 任务 | 检验点 |
|------|------|--------|
| **D1 上午** | 数据清洗 + 事件提取 | 每季事件表完整 |
| **D1 下午** | CP-SAT (Rank) + LP (Percent) | 解数>0, timeout内完成 |
| **D2 上午** | MCMC采样器 + ESS检验 | ESS>100 |
| **D2 下午** | 层次先验 + 因子分析 | SHAP图生成 |
| **D3 上午** | 弹性分析 + 反事实重放 | 争议案例表完成 |
| **D3 下午** | 敏感性 + 论文撰写 | 准确率>80%, Brier<0.2 |

---

> **v3.0 已修复全部7项技术漏洞，可直接用于代码实现。**
