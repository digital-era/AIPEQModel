import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')

# ==========================================
# 0. 检查数据源
# ==========================================
if 'df_combined' not in locals():
    raise ValueError("请确保 'df_combined' 变量已经加载到内存中！")

print("数据加载成功，原始数据形状:", df_combined.shape)

# 为了保护原始数据，拷贝一份进行操作
df = df_combined.copy()

# ==========================================
# 1. 数据预处理
# ==========================================

# 确保日期格式正确
df['日期'] = pd.to_datetime(df['日期'])

# 确保数据按【代码】和【日期】排序，这是计算时间序列特征的基础
df.sort_values(by=['代码', '日期'], ascending=[True, True], inplace=True)

# 定义需要分析的特征列
feature_cols = [
    'PotScore', 
    '总净流入占比_5日总和',
    '主力净流入-净占比', 
    '超大单净流入-净占比'
]

# ==========================================
# 2. 特征工程：构建3天观察窗口 & 预测目标
# ==========================================

print("正在构建3日窗口特征和目标变量...")

# A. 构建观察窗口特征 (X)
# 逻辑：计算各特征在过去3天（包含当天）的移动平均值
# 这能反映资金在3天窗口内的持续意图，比单日数据更稳健
for col in feature_cols:
    # transform保留原行数，方便后续赋值
    df[f'{col}_3日均值'] = df.groupby('代码')[col].transform(lambda x: x.rolling(window=3).mean())

# B. 构建预测目标 (Y)
# 逻辑：推测第5天收盘 vs 第3天收盘
# 当前行索引为 T (第3天)，我们需要 T+2 (第5天) 的收盘价
# shift(-2) 表示将未来第2行的数据向上平移对齐到当前行
df['Target_第5天收益率'] = df.groupby('代码')['收盘价'].shift(-2) / df['收盘价'] - 1

# C. 清洗数据
# 1. 去除Rolling产生的NaN (每只股票前2天无法计算3日均值)
# 2. 去除Shift产生的NaN (每只股票最后2天没有未来的第5天数据)
df_model = df.dropna().copy()

print(f"清洗后用于建模的数据量: {len(df_model)} 条")

# ==========================================
# 3. 相关性分析 (线性关系)
# ==========================================

# 准备用于分析的列（原始特征 + 3日均值特征）
analysis_features = feature_cols + [f'{col}_3日均值' for col in feature_cols]

# 计算相关系数矩阵
corr_matrix = df_model[analysis_features + ['Target_第5天收益率']].corr()
target_corr = corr_matrix['Target_第5天收益率'].drop('Target_第5天收益率').sort_values(ascending=False)

print("\n=== 各特征与【第5天收益率】的皮尔逊相关系数 ===")
print(target_corr)

# ==========================================
# 4. 随机森林建模 (挖掘非线性权重)
# ==========================================

X = df_model[analysis_features]
y = df_model['Target_第5天收益率']

# 划分训练集和测试集 (不打乱顺序，模拟时间序列回测)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 初始化模型
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1)

# 训练
rf_model.fit(X_train, y_train)

# 预测与评估
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n模型拟合优度 (R2 Score): {r2:.4f} (注：金融数据通常较低，正数即代表有预测力)")

# ==========================================
# 5. 结果可视化与分析
# ==========================================

# 提取特征重要性
importances = pd.DataFrame({
    'Feature': analysis_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 映射英文名以解决Colab绘图中文乱码问题
col_mapping = {
    'PotScore': 'PotScore',
    'PotScore_3日均值': 'PotScore_3d_Avg',
    '总净流入占比_5日总和': 'Inflow_5d_Sum',
    '总净流入占比_5日总和_3日均值': 'Inflow_5d_Sum_3d_Avg',
    '主力净流入-净占比': 'Main_Inflow_Pct',
    '主力净流入-净占比_3日均值': 'Main_Inflow_Pct_3d_Avg',
    '超大单净流入-净占比': 'Super_Inflow_Pct',
    '超大单净流入-净占比_3日均值': 'Super_Inflow_Pct_3d_Avg'
}

importances['Feature_En'] = importances['Feature'].map(col_mapping)

# 打印重要性排行
print("\n=== 特征重要性排行 (基于随机森林) ===")
print(importances[['Feature', 'Importance']])

# 绘图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature_En', data=importances, palette='viridis')
plt.title('Feature Importance (Predicting Day 5 Return vs Day 3)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# ==========================================
# 6. 策略解读
# ==========================================
top_feat = importances.iloc[0]['Feature']
top_corr_val = target_corr[top_feat]
relation = "正相关 (越大越好)" if top_corr_val > 0 else "负相关 (越小越好)"

print("\n" + "="*30)
print("   量化分析结论")
print("="*30)
print(f"1. 最核心的预测指标是：[{top_feat}]")
print(f"2. 指标方向性：该指标与后市涨幅呈 {relation}。")
print(f"3. 观察窗口验证：")
if '3日均值' in top_feat:
    print("   -> 结果显示【3日均值】特征比单日特征更重要。")
    print("   -> 建议：不要只看当天的资金流，要看过去3天的平均水平，持续的资金行为更有效。")
else:
    print("   -> 结果显示【单日爆发】特征更重要，短期爆发力可能比持续性更关键。")




# ==========================================
# 7. 生成明日实盘推荐 (Top 4 组合)
# ==========================================

print("正在计算最新市场数据的预测得分...")

# 1. 获取最新的数据切片 (T日)
latest_date = df['日期'].max()
print(f"最新观测日期 (T日): {latest_date}")

# 提取最新一天的所有标的数据作为候选池
current_data = df[df['日期'] == latest_date].copy()

# ==========================================
# [新增] 策略过滤：剔除3日内单日涨幅超3%的标的
# ==========================================
print("-" * 30)
print("正在执行策略过滤：【观察期3日内，单日涨幅不超过3%】...")

# 1. 确定观察窗口（最近3个交易日）
all_dates = sorted(df['日期'].unique())
obs_dates = all_dates[-3:]
print(f"观察窗口: {[d.strftime('%Y-%m-%d') for d in obs_dates]}")

# 2. 提取候选标的在观察窗口内的历史数据
# 注意：这里假设数据中包含 '涨跌幅' 列（单位通常为%，如 2.5 代表 2.5%）
# 如果您的列名不同（例如 'pct_chg'），请修改下面的 col_name
col_name_pct = '涨跌幅' 

if col_name_pct not in df.columns:
    raise ValueError(f"数据中缺少 '{col_name_pct}' 列，无法进行涨幅筛选，请检查列名。")

# 提取近3日数据
history_3d = df[df['日期'].isin(obs_dates) & df['代码'].isin(current_data['代码'])]

# 3. 计算每个标的在近3日的最大单日涨幅
max_pct_3d = history_3d.groupby('代码')[col_name_pct].max()

# 4. 筛选符合条件（最大涨幅 <= 3.0）的标的
# 注意：这里设定阈值为 3.0 (即3%)。如果您的数据是 0.03 格式，请改为 0.03
valid_codes = max_pct_3d[max_pct_3d <= 3.0].index

# 5. 更新候选池
original_count = len(current_data)
current_data = current_data[current_data['代码'].isin(valid_codes)]
filtered_count = len(current_data)

print(f"筛选结果: {original_count} -> {filtered_count} 只 (剔除 {original_count - filtered_count} 只已启动标的)")
print("-" * 30)

# ==========================================
# 继续原有预测逻辑
# ==========================================

# 2. 检查特征完整性
X_live = current_data[analysis_features]

# 检查空值
if X_live.isnull().values.any():
    print("警告：部分标的因特征数据不足将被剔除。")
    current_data = current_data.dropna(subset=analysis_features)
    X_live = current_data[analysis_features]

# 3. 模型预测 (T+2 潜力)
if len(current_data) == 0:
    print("错误：经过筛选后没有剩余标的，无法推荐。")
else:
    current_data['预测得分'] = rf_model.predict(X_live)

    # 4. 选股策略 (Top 4)
    top_picks = current_data.sort_values(by='预测得分', ascending=False).head(4)

    # 5. 权重分配
    if len(top_picks) < 1:
        print("警告：没有足够的标的进行推荐。")
    else:
        scores = top_picks['预测得分'].values
        min_score = scores.min()
        if min_score < 0:
            adjusted_scores = scores - min_score + 0.01 
        else:
            adjusted_scores = scores

        weights = adjusted_scores / adjusted_scores.sum() * 100
        top_picks['推荐权重(%)'] = weights.round(2)

        # ==========================================
        # 8. 输出推荐结果 (列名已根据特征分析优化)
        # ==========================================
        print("\n" + "#"*50)
        print(f"   🚀 明日 ({latest_date.date()}) 潜伏组合推荐")
        print(f"   (筛选标准：预测得分高 + 近3日未曾大涨)")
        print("#"*50)

        # 定义展示的核心特征 (基于之前的分析结论)
        core_feature = '超大单净流入-净占比' # 排名 No.1
        sec_feature = '主力净流入-净占比'   # 排名 No.2

        output_cols = ['代码', '名称', '收盘价', '涨跌幅', '预测得分', '推荐权重(%)', core_feature, sec_feature]
        
        # 确保列存在，防止报错
        existing_cols = [c for c in output_cols if c in top_picks.columns]
        result_df = top_picks[existing_cols].copy()

        # 重命名列名以便阅读
        rename_dict = {
            core_feature: '核心(超大单占比)',
            sec_feature: '辅助(主力占比)',
            '涨跌幅': 'T日涨幅(%)'
        }
        result_df.rename(columns=rename_dict, inplace=True)

        # 打印表格
        print(result_df.to_markdown(index=False))

        print("\n【策略逻辑说明】")
        print("1. 基础筛选：剔除近3日内有过单日涨幅>3%的标的，确保介入位置处于相对低位/蓄势期。")
        print(f"2. 核心驱动：基于[{core_feature}]（重要性0.35）预测爆发潜力。")
        print("3. 权重分配：根据随机森林预测得分动态分配仓位。")

        # 可视化
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 6))
        # 字体设置防止乱码
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.pie(top_picks['推荐权重(%)'], labels=top_picks['名称'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
        plt.title(f"潜伏策略组合权重 ({latest_date.date()})")
        plt.show()
