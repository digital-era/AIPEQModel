数据框df_combined为实际数据，包括如下列
Index(['代码', '名称', 'PotScore', '总净流入占比_5日总和', '日期', '收盘价', '涨跌幅', '主力净流入-净额',
       '主力净流入-净占比', '超大单净流入-净额', '超大单净流入-净占比', '大单净流入-净额', '大单净流入-净占比',
       '中单净流入-净额', '中单净流入-净占比', '小单净流入-净额', '小单净流入-净占比', '总净流入占比', 'UDI_Close',
       'UDI_Percent', 'l2name'],
      dtype='object')

目前已探测到包括两个重要流形：超大单净流入-净占比 & PostScore
请通过实际数据学习P大幅连续上涨之前 如何捕获超大单净流入-净占比、PostScore与收盘价量化信号

这是一个非常典型的量化特征挖掘问题。既然你已经锁定了两个关键特征（流形）——`PotScore` 和 `超大单净流入-净占比`，我们的目标是在高维空间中找到这两个特征与“未来股价大幅上涨”之间的映射关系。

为了“学习”由于这两个特征引发的上涨模式，我们需要执行以下三个步骤：

1.  **数据标注 (Labeling)**：定义什么是“大幅连续上涨”。
2.  **模式识别 (Pattern Recognition)**：观察在上涨启动前，$T-N$天这两个指标的形态（是突增、背离还是高位震荡？）。
3.  **策略回测 (Signal Logic)**：构建具体的量化过滤条件。

请在你的Python环境中运行以下代码。这段代码是专门为你提供的数据结构设计的。

### 第一步：数据预处理与目标定义

首先，我们需要清洗数据并计算“未来收益”，以便机器知道我们想捕获什么样的行情。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 df_combined 已经加载
df = df_combined.copy()

# 1. 基础处理
df['日期'] = pd.to_datetime(df['日期'])
df.sort_values(by=['代码', '日期'], inplace=True)

# 2. 定义“大幅连续上涨” (Target)
# 这里的逻辑是：计算未来 N 天的最大累计涨幅。
# 比如：未来10天内，最高价相对于当前收盘价的涨幅。
N_DAYS = 10
THRESHOLD = 0.20  # 定义大幅上涨为：未来10天内最大涨幅超过20%

def calculate_forward_max_return(group):
    # 下一天的开盘价（作为买入价）
    buy_price = group['收盘价'].shift(-1)
    # 未来N天的最高收盘价（简化计算，用收盘价代替最高价，或者你有最高价字段更好）
    # rolling min_periods=1 保证数据边缘也能算
    future_max = group['收盘价'].rolling(window=N_DAYS, min_periods=1).max().shift(-N_DAYS)
    return (future_max / buy_price) - 1

df['future_max_ret'] = df.groupby('代码', group_keys=False).apply(calculate_forward_max_return)

# 标记正样本：未来确实大涨了
df['is_big_pump'] = df['future_max_ret'] > THRESHOLD

# 3. 特征平滑 (去噪)
# 超大单资金流通常波动很大，建议看3日或5日均值，或者累积值
df['super_flow_ma3'] = df.groupby('代码')['超大单净流入-净占比'].transform(lambda x: x.rolling(3).mean())
df['pot_score_ma3'] = df.groupby('代码')['PotScore'].transform(lambda x: x.rolling(3).mean())

print(f"正样本数量: {df['is_big_pump'].sum()} / {len(df)}")
```

---

### 第二步：可视化探测“流形”区域 (Manifold Detection)

我们需要在二维平面上（X轴：PotScore，Y轴：超大单占比）画出热力图，看看**在哪一个区域（坐标区间）**，未来大涨的概率最高。

```python
def plot_signal_manifold(df):
    # 去除空值
    plot_data = df.dropna(subset=['PotScore', '超大单净流入-净占比', 'future_max_ret'])
    
    # 限制一下显示范围，去掉极值影响绘图（可选）
    # plot_data = plot_data[(plot_data['PotScore'] > 0) & (plot_data['超大单净流入-净占比'] > -5)]

    plt.figure(figsize=(12, 5))
    
    # 图1：散点图 - 红色代表未来大涨，蓝色代表没涨
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=plot_data, 
        x='PotScore', 
        y='超大单净流入-净占比', 
        hue='future_max_ret',
        palette='coolwarm',
        alpha=0.6,
        size='future_max_ret',
        sizes=(10, 100)
    )
    plt.title('PotScore vs SuperFlow (Color=Future Return)')
    plt.axhline(0, color='grey', linestyle='--')
    
    # 图2：分箱热力图 (寻找高胜率区间)
    # 将两个指标离散化为箱体
    plt.subplot(1, 2, 2)
    df_bin = plot_data.copy()
    df_bin['pot_bin'] = pd.qcut(df_bin['PotScore'], q=10, labels=False, duplicates='drop')
    df_bin['flow_bin'] = pd.qcut(df_bin['超大单净流入-净占比'], q=10, labels=False, duplicates='drop')
    
    pivot_win_rate = df_bin.pivot_table(
        index='flow_bin', 
        columns='pot_bin', 
        values='is_big_pump', 
        aggfunc='mean' # 计算出现大涨的概率
    )
    
    sns.heatmap(pivot_win_rate, cmap='YlOrRd', annot=True, fmt=".2f")
    plt.title('Win Rate Heatmap (X: PotScore Rank, Y: Flow Rank)')
    plt.gca().invert_yaxis() # 让Y轴高rank在上面
    
    plt.tight_layout()
    plt.show()

plot_signal_manifold(df)
```

**如何解读结果：**
*   如果不通过运行，通常的规律是：**右上角**（高PotScore + 高超大单流入）胜率最高。
*   但更具价值的是发现**右下角或中间偏右**（PotScore极高，但超大单刚开始流入，甚至微幅流出时）是否潜伏着“底部反转”信号。

---

### 第三步：事件研究 (Event Study) - 爆发前发生了什么？

我们要把所有“大涨前夜”的数据对齐，观察 T-5 到 T-1 天，这两个指标的走势。

```python
def event_study_analysis(df):
    # 提取爆发日 (T=0)，即 is_big_pump 为 True 且 前一天为 False (或者简单取满足条件的日子)
    # 这里简单取所有满足未来大涨的日子作为观察点
    events = df[df['is_big_pump'] == True].index
    
    # 收集爆发前5天到爆发后3天的数据轨迹
    trajectories = []
    
    for idx in events:
        # 获取当前行位置
        loc = df.index.get_loc(idx)
        if loc < 5 or loc > len(df) - 5: continue
        
        # 截取切片
        subset = df.iloc[loc-5 : loc+1].copy() # T-5 到 T0
        
        # 归一化处理以便聚合 (以T-5的值为基准，或者不做归一化直接看绝对值)
        # 这里看绝对值的均值趋势
        trajectories.append({
            'days': range(-5, 1),
            'PotScore': subset['PotScore'].values,
            'SuperFlow': subset['超大单净流入-净占比'].values,
            'Close': subset['收盘价'].values / subset['收盘价'].values[0] # 价格归一化
        })
    
    # 聚合绘图
    res_df = pd.DataFrame()
    for t in trajectories:
        temp = pd.DataFrame(t)
        res_df = pd.concat([res_df, temp])
        
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    sns.lineplot(data=res_df, x='days', y='Close')
    plt.title('Price Trend before Big Pump (Normalized)')
    
    plt.subplot(3, 1, 2)
    sns.lineplot(data=res_df, x='days', y='SuperFlow')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Super Large Order Flow Trend')
    
    plt.subplot(3, 1, 3)
    sns.lineplot(data=res_df, x='days', y='PotScore')
    plt.title('PotScore Trend')
    
    plt.tight_layout()
    plt.show()

event_study_analysis(df)
```

---

### 第四步：构建量化信号捕获器

基于上述分析（假设通常规律：PotScore高位企稳 + 超大单异动），我们可以构建一个具体的筛选策略。

**核心逻辑假设（共振模型）：**
1.  **PotScore** 代表个股的“势能”或“评分”，需要处于高位（例如 > 80 或 Top 20%）。
2.  **超大单** 代表“主力意志”，需要出现明显的净流入，或者相比过去5天有剧烈提升。
3.  **收盘价** 处于相对低位或刚突破（避免追高）。

```python
def capture_signals(df):
    # 1. 动态阈值计算
    # PotScore 应该大于过去 20 天的 75% 分位数 (保持高位)
    df['pot_quantile'] = df.groupby('代码')['PotScore'].transform(lambda x: x.rolling(20).quantile(0.75))
    
    # 超大单 必须是正流入，且大于过去5天均值 (突增)
    df['flow_ma5'] = df.groupby('代码')['超大单净流入-净占比'].transform(lambda x: x.rolling(5).mean())
    
    # 2. 信号定义
    # 条件A: PotScore 足够高 (说明基本面或技术面评分好)
    cond_pot = df['PotScore'] > 50  # 这里的50需要根据你的PotScore实际分布调整，如果是0-100分制
    
    # 条件B: 主力资金大幅流入 (大于某个绝对阈值 或 突增)
    # 假设 '超大单净流入-净占比' 单位是百分比，比如 5 代表 5%
    cond_flow_spike = (df['超大单净流入-净占比'] > 3) & (df['超大单净流入-净占比'] > df['flow_ma5'] * 1.5)
    
    # 条件C: 价格形态 (量价配合)
    # 当天涨幅不能过大（比如已经涨停了就不追了），也不能大跌
    cond_price = (df['涨跌幅'] > 0) & (df['涨跌幅'] < 9.5)
    
    # 组合信号
    df['SIGNAL_BUY'] = cond_pot & cond_flow_spike & cond_price
    
    return df

# 应用并查看结果
df_result = capture_signals(df)

# 查看捕获到的信号，其未来的表现
signal_performance = df_result[df_result['SIGNAL_BUY'] == True]['future_max_ret'].describe()
print("\n捕获信号后的未来10天最大涨幅统计：")
print(signal_performance)

# 输出最近出现信号的股票
recent_signals = df_result[df_result['SIGNAL_BUY'] == True].sort_values('日期', ascending=False).head(10)
print("\n最近捕获的信号案例：")
print(recent_signals[['日期', '代码', '名称', '收盘价', 'PotScore', '超大单净流入-净占比']])
```

### 总结与建议

根据经验，要在实际数据中捕获这两个流形（Manifold）的爆发点，请重点关注以下三种具体的**微观结构**：

1.  **“主力潜伏”模式 (Divergence)**：
    *   **现象**：股价（收盘价）横盘震荡或微跌，但`超大单净流入-净占比`连续3-5天为正，且`PotScore`缓慢爬升。
    *   **含义**：主力在压价吸筹，评分变好说明技术面正在修复。
    *   **捕获逻辑**：`Correlation(Price, Flow) < 0` (过去5天) AND `Sum(Flow) > Threshold`。

2.  **“空中加油”模式 (Resonance)**：
    *   **现象**：`PotScore` 长期维持高位（>80），股价经历一波回调后，`超大单净流入`突然出现一根巨大的阳线（Spike）。
    *   **含义**：强势股回调结束，主力再次发力。
    *   **捕获逻辑**：`PotScore > 80` AND `Flow > 2*Std_Dev(Flow_20days)`。

3.  **“反转启动”模式 (Reversal)**：
    *   **现象**：`PotScore` 从低位急速拉升穿越中轴，伴随`超大单净流入`由负转正。
    *   **含义**：情绪与资金的双重反转。

你可以用上面的代码，通过调整 `cond_pot` 和 `cond_flow_spike` 里的阈值，来逼近这三种模式。建议优先观察**“主力潜伏”模式**，其风险收益比通常最好。
