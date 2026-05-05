import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load dataset
df = pd.read_csv('all_seasons.csv', index_col=0)

print("Dataset loaded")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")

# intial inspection
print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
display(df.head())

print("\n--- Descriptive Statistics ---")
display(df.describe())

print("\n--- Missing values ---")
print(df.isnull().sum())


# drop the unamed index coulumn
df.drop(columns=['college'], inplace=True)
print("\n Dropped 'college' column (too many missing values, irrelevant to RQs)")


# fix datatypes
# draft_round and draft_number contain 'Undrafted' strings → handle before converting
# Replace 'Undrafted' with NaN, then convert to numeric
df['draft_round'] = pd.to_numeric(df['draft_round'].replace('Undrafted', np.nan), errors='coerce')
df['draft_number'] = pd.to_numeric(df['draft_number'].replace('Undrafted', np.nan), errors='coerce')
df['draft_year']  = pd.to_numeric(df['draft_year'].replace('Undrafted', np.nan), errors='coerce')

# age -> ensure it's int-compatible
df['age'] = df['age'].astype(int)

print("\n Fixed data types")
print(df[['draft_round', 'draft_number', 'draft_year', 'age']].dtypes)

# Remove duplicate player-season records
before = len(df)
df.drop_duplicates(subset=['player_name', 'season'], inplace=True)
after = len(df)
print(f"\n Removed {before - after} duplicate player-season rows")
print(f"   Remaining rows: {after:,}")

# Filter out players with very few games played
# Players with < 10 games are statistical noise; filter them out.
before = len(df)
df = df[df['gp'] >= 10]
after = len(df)
print(f"\n Removed {before - after} rows where games played (gp) < 10")
print(f"   Remaining rows: {after:,}")

# Compute BMI column
# BMI = weight(kg) / height(m)^2
# player_height is in cm, player_weight is in kg
df['bmi'] = df['player_weight'] / ((df['player_height'] / 100) ** 2)
print("\n Added 'bmi' column")
print(df['bmi'].describe().round(2))


# Add 'age_group' column (used in EDA)
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 25, 30, 100],
                         labels=['18-25', '26-30', '31+'])
print("\n Added 'age_group' column")
print(df['age_group'].value_counts())


# Add 'scorer_group' column
# High scorers: pts >= 20 PPG | Low scorers: pts < 10 PPG
# (We exclude the middle group 10–19 PPG for a clean two-group t-test)
df['scorer_group'] = np.where(df['pts'] >= 20, 'High (≥20 PPG)',
                     np.where(df['pts'] <  10, 'Low (<10 PPG)', 'Mid'))
print("\n Added 'scorer_group' column")
print(df['scorer_group'].value_counts())


# Final cleaned dataset summary
print("\n--- FINAL CLEANED DATASET ---")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print("\nNull values remaining:")
print(df.isnull().sum()[df.isnull().sum() > 0])

display(df.head())


# STEP 11: Save cleaned CSV
df.to_csv('all_seasons_cleaned.csv', index=False)
print("\n Saved cleaned dataset as 'all_seasons_cleaned.csv'")


# SUBSETS READY FOR HYPOTHESIS TESTING

# BMI - High vs Low scorers
high_scorers = df[df['scorer_group'] == 'High (≥20 PPG)']['bmi'].dropna()
low_scorers  = df[df['scorer_group'] == 'Low (<10 PPG)']['bmi'].dropna()

print("\n--- RQ2 Subset Sizes ---")
print(f"High scorers (≥20 PPG): n = {len(high_scorers)}")
print(f"Low scorers (<10 PPG):  n = {len(low_scorers)}")
print(f"\nHigh scorers BMI  →  mean={high_scorers.mean():.2f}, std={high_scorers.std():.2f}")
print(f"Low scorers  BMI  →  mean={low_scorers.mean():.2f}, std={low_scorers.std():.2f}")

# Hypothesis Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, levene, shapiro, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Load cleaned dataset
df = pd.read_csv('all_seasons_cleaned.csv')
print(f"Dataset loaded: {df.shape[0]:,} rows\n")

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# RESEARCH QUESTION 1
# Are players in the tallest quartile significantly taller than players in the shortest quartile?
# Test: One-tailed Welch's t-test

print_section("RQ1: Tall vs Short Players — One-Tailed t-Test")

tall_threshold  = df['player_height'].quantile(0.75)
short_threshold = df['player_height'].quantile(0.25)

tall  = df[df['player_height'] >= tall_threshold]['player_height'].dropna()
short = df[df['player_height'] <= short_threshold]['player_height'].dropna()

print(f"Tall  players (≥ {tall_threshold:.1f} cm): n={len(tall)},  mean={tall.mean():.2f} cm, std={tall.std():.2f}")
print(f"Short players (≤ {short_threshold:.1f} cm): n={len(short)}, mean={short.mean():.2f} cm, std={short.std():.2f}")

# Hypotheses
print("\nH₀: μ_tall ≤ μ_short  (tall players are NOT significantly taller)")
print("H₁: μ_tall > μ_short  (tall players ARE significantly taller)")
print("α  = 0.05 | One-tailed test")

# Assumption check - Levene's test for equal variances
lev_stat, lev_p = levene(tall, short)
print(f"\nLevene's Test: stat={lev_stat:.4f}, p={lev_p:.4f}")
print("→ Using Welch's t-test (does not assume equal variances)" if lev_p < 0.05
      else "→ Variances appear equal, but Welch's t-test is still robust")

# Welch's t-test (one-tailed: alternative='greater')
t_stat, p_two = ttest_ind(tall, short, equal_var=False)
p_one = p_two / 2   # convert to one-tailed p-value

# Degrees of freedom (Welch-Satterthwaite)
n1, s1 = len(tall),  tall.std()
n2, s2 = len(short), short.std()
df_welch = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

# 95% Confidence Interval for difference in means
se = np.sqrt(s1**2/n1 + s2**2/n2)
t_crit = stats.t.ppf(0.95, df=df_welch)   # one-tailed critical value
diff = tall.mean() - short.mean()
ci_lower = diff - t_crit * se
ci_upper = diff + t_crit * se

# Cohen's d
pooled_std = np.sqrt((s1**2 + s2**2) / 2)
cohens_d   = diff / pooled_std

print(f"\n--- Results ---")
print(f"t-statistic : {t_stat:.4f}")
print(f"Degrees of freedom (Welch): {df_welch:.1f}")
print(f"p-value (one-tailed) : {p_one:.6f}")
print(f"t-critical (α=0.05, one-tailed): {t_crit:.4f}")
print(f"95% CI for (μ_tall − μ_short): ({ci_lower:.2f}, {ci_upper:.2f}) cm")
print(f"Cohen's d : {cohens_d:.4f}  → {'Small' if abs(cohens_d)<0.5 else 'Medium' if abs(cohens_d)<0.8 else 'Large'} effect")

print(f"\n--- Decision ---")
if p_one < 0.05:
    print(f"p={p_one:.6f} < 0.05 → REJECT H₀")
    print("Conclusion: Tall-quartile players are significantly taller than short-quartile players.")
else:
    print(f"p={p_one:.6f} ≥ 0.05 → FAIL TO REJECT H₀")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('RQ1: Height Distribution — Tall vs Short Quartile Players', fontsize=13, fontweight='bold')

axes[0].hist(short, bins=25, alpha=0.6, color='steelblue', label=f'Short (≤{short_threshold:.0f}cm)')
axes[0].hist(tall,  bins=25, alpha=0.6, color='tomato',    label=f'Tall  (≥{tall_threshold:.0f}cm)')
axes[0].axvline(short.mean(), color='steelblue', linestyle='--', linewidth=1.5)
axes[0].axvline(tall.mean(),  color='tomato',    linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Height (cm)')
axes[0].set_ylabel('Count')
axes[0].set_title('Height Distributions')
axes[0].legend()

axes[1].boxplot([short, tall], labels=[f'Short\n(≤{short_threshold:.0f}cm)', f'Tall\n(≥{tall_threshold:.0f}cm)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_ylabel('Height (cm)')
axes[1].set_title(f'Boxplot  |  t={t_stat:.2f}, p={p_one:.4f}')
plt.tight_layout()
plt.savefig('rq1_height.png', dpi=150, bbox_inches='tight')
plt.show()
print(" RQ1 plot saved as rq1_height.png")

# RESEARCH QUESTION 2
# Does BMI differ between high scorers (≥20 PPG) and
# low scorers (<10 PPG)?
# Test: Two-tailed Welch's t-test
print_section("RQ2: BMI — High vs Low Scorers — Two-Tailed t-Test")

high = df[df['scorer_group'] == 'High (≥20 PPG)']['bmi'].dropna()
low  = df[df['scorer_group'] == 'Low (<10 PPG)']['bmi'].dropna()

print(f"High scorers (≥20 PPG): n={len(high)}, BMI mean={high.mean():.4f}, std={high.std():.4f}")
print(f"Low  scorers (<10 PPG): n={len(low)},  BMI mean={low.mean():.4f}, std={low.std():.4f}")

print("\nH₀: μ_high_BMI = μ_low_BMI  (no difference in BMI between groups)")
print("H₁: μ_high_BMI ≠ μ_low_BMI  (BMI differs between high and low scorers)")
print("α  = 0.05 | Two-tailed test")

# Levene's test
lev_stat2, lev_p2 = levene(high, low)
print(f"\nLevene's Test: stat={lev_stat2:.4f}, p={lev_p2:.4f}")

# Welch's t-test (two-tailed)
t2, p2 = ttest_ind(high, low, equal_var=False)

# Degrees of freedom
n1h, s1h = len(high), high.std()
n2l, s2l = len(low),  low.std()
df_welch2 = (s1h**2/n1h + s2l**2/n2l)**2 / ((s1h**2/n1h)**2/(n1h-1) + (s2l**2/n2l)**2/(n2l-1))

# 95% CI (two-tailed)
se2       = np.sqrt(s1h**2/n1h + s2l**2/n2l)
t_crit2   = stats.t.ppf(0.975, df=df_welch2)
diff2     = high.mean() - low.mean()
ci2_lower = diff2 - t_crit2 * se2
ci2_upper = diff2 + t_crit2 * se2

# Cohen's d
pooled_std2 = np.sqrt((s1h**2 + s2l**2) / 2)
cohens_d2   = diff2 / pooled_std2

print(f"\n--- Results ---")
print(f"t-statistic : {t2:.4f}")
print(f"Degrees of freedom (Welch): {df_welch2:.1f}")
print(f"p-value (two-tailed) : {p2:.6f}")
print(f"t-critical (α=0.05, two-tailed): ±{t_crit2:.4f}")
print(f"95% CI for (μ_high − μ_low): ({ci2_lower:.4f}, {ci2_upper:.4f})")
print(f"Cohen's d : {cohens_d2:.4f}  → {'Small' if abs(cohens_d2)<0.5 else 'Medium' if abs(cohens_d2)<0.8 else 'Large'} effect")

print(f"\n--- Decision ---")
if p2 < 0.05:
    print(f"p={p2:.6f} < 0.05 → REJECT H₀")
    print("Conclusion: BMI is significantly different between high and low scorers.")
else:
    print(f"p={p2:.6f} ≥ 0.05 → FAIL TO REJECT H₀")
    print("Conclusion: No significant difference in BMI between high and low scorers.")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('RQ2: BMI — High Scorers (≥20 PPG) vs Low Scorers (<10 PPG)', fontsize=13, fontweight='bold')

axes[0].hist(low,  bins=30, alpha=0.6, color='steelblue', label='Low (<10 PPG)')
axes[0].hist(high, bins=30, alpha=0.6, color='tomato',    label='High (≥20 PPG)')
axes[0].axvline(low.mean(),  color='steelblue', linestyle='--', linewidth=1.5)
axes[0].axvline(high.mean(), color='tomato',    linestyle='--', linewidth=1.5)
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Count')
axes[0].set_title('BMI Distributions')
axes[0].legend()

axes[1].boxplot([low, high], labels=['Low (<10 PPG)', 'High (≥20 PPG)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightyellow'))
axes[1].set_ylabel('BMI')
axes[1].set_title(f'Boxplot  |  t={t2:.2f}, p={p2:.4f}')
plt.tight_layout()
plt.savefig('rq2_bmi.png', dpi=150, bbox_inches='tight')
plt.show()
print(" RQ2 plot saved as rq2_bmi.png")

# RESEARCH QUESTION 3
# Does points per game differ across age groups
# (18-25, 26-30, 31+)?
# Test: One-way ANOVA + Tukey HSD post-hoc
print_section("RQ3: PPG Across Age Groups — One-Way ANOVA")

g1 = df[df['age_group'] == '18-25']['pts'].dropna()
g2 = df[df['age_group'] == '26-30']['pts'].dropna()
g3 = df[df['age_group'] == '31+']['pts'].dropna()

print(f"18-25 : n={len(g1)}, mean={g1.mean():.4f}, std={g1.std():.4f}")
print(f"26-30 : n={len(g2)}, mean={g2.mean():.4f}, std={g2.std():.4f}")
print(f"31+   : n={len(g3)}, mean={g3.mean():.4f}, std={g3.std():.4f}")

print("\nH₀: μ_18-25 = μ_26-30 = μ_31+  (PPG does not differ across age groups)")
print("H₁: At least one age group has a different mean PPG")
print("α  = 0.05")

# Manual ANOVA calculations
all_groups  = [g1, g2, g3]
group_names = ['18-25', '26-30', '31+']
k           = len(all_groups)                      # number of groups
N           = sum(len(g) for g in all_groups)      # total observations
grand_mean  = np.concatenate(all_groups).mean()

# SS Between (variation due to group differences)
SS_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in all_groups)

# SS Within (variation within each group)
SS_within  = sum(((g - g.mean())**2).sum() for g in all_groups)

# SS Total
SS_total   = SS_between + SS_within

# Degrees of freedom
df_between = k - 1
df_within  = N - k

# Mean Squares
MS_between = SS_between / df_between
MS_within  = SS_within  / df_within

# F-statistic
F_stat = MS_between / MS_within

# p-value
p_anova = 1 - stats.f.cdf(F_stat, df_between, df_within)

# F-critical
F_crit = stats.f.ppf(0.95, df_between, df_within)

# Eta-squared (effect size)
eta_sq = SS_between / SS_total

print(f"\n--- Manual ANOVA Table ---")
print(f"{'Source':<12} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p':>10}")
print("-"*64)
print(f"{'Between':<12} {SS_between:>12.4f} {df_between:>6} {MS_between:>12.4f} {F_stat:>10.4f} {p_anova:>10.6f}")
print(f"{'Within':<12} {SS_within:>12.4f} {df_within:>6} {MS_within:>12.4f}")
print(f"{'Total':<12} {SS_total:>12.4f} {N-1:>6}")
print(f"\nF-critical (α=0.05, df1={df_between}, df2={df_within}): {F_crit:.4f}")
print(f"Eta-squared (η²) : {eta_sq:.4f}  → {'Small' if eta_sq<0.06 else 'Medium' if eta_sq<0.14 else 'Large'} effect")

print(f"\n--- Decision ---")
if p_anova < 0.05:
    print(f"F={F_stat:.4f} > F_crit={F_crit:.4f}, p={p_anova:.6f} < 0.05 → REJECT H₀")
    print("Conclusion: PPG differs significantly across at least one age group.")
else:
    print(f"F={F_stat:.4f} ≤ F_crit={F_crit:.4f}, p={p_anova:.6f} ≥ 0.05 → FAIL TO REJECT H₀")

# Tukey HSD Post-Hoc
print("\n--- Tukey HSD Post-Hoc Test (which pairs differ?) ---")

from itertools import combinations

q_crit = 3.31   # Studentized range q-statistic for k=3, α=0.05 (large df_within)
HSD    = q_crit * np.sqrt(MS_within / min(len(g) for g in all_groups))

print(f"HSD threshold = {HSD:.4f}")
print(f"\n{'Pair':<20} {'Diff':>10} {'Significant?':>15}")
print("-"*48)
for (i, gi), (j, gj) in combinations(enumerate(all_groups), 2):
    diff_pair = abs(gi.mean() - gj.mean())
    sig       = "YES ✓" if diff_pair > HSD else "NO"
    print(f"{group_names[i]} vs {group_names[j]:<10} {diff_pair:>10.4f} {sig:>15}")

# Visualization - 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('RQ3: Points Per Game Across Age Groups', fontsize=13, fontweight='bold')

# (a) Boxplot
data_plot   = [g1, g2, g3]
bp = axes[0].boxplot(data_plot, labels=group_names, patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
axes[0].set_ylabel('Points per Game')
axes[0].set_title(f'Boxplot by Age Group\nF={F_stat:.2f}, p={p_anova:.4f}')

# (b) Bar chart of means with error bars
means = [g.mean() for g in all_groups]
stds  = [g.std()  for g in all_groups]
colors = ['#5B9BD5', '#ED7D31', '#70AD47']
axes[1].bar(group_names, means, color=colors, alpha=0.8,
            yerr=[s/np.sqrt(len(g)) for s, g in zip(stds, all_groups)],
            capsize=5, error_kw=dict(elinewidth=1.5))
axes[1].set_ylabel('Mean PPG')
axes[1].set_title('Mean PPG with 95% SE Error Bars')
for i, (m, g) in enumerate(zip(means, all_groups)):
    axes[1].text(i, m + 0.2, f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')

# (c) ANOVA variance partition pie chart
axes[2].pie([SS_between, SS_within],
            labels=[f'Between Groups\n(SS={SS_between:.0f})',
                    f'Within Groups\n(SS={SS_within:.0f})'],
            colors=['#ED7D31', '#5B9BD5'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10})
axes[2].set_title(f'Variance Partition\nη²={eta_sq:.4f}')

plt.tight_layout()
plt.savefig('rq3_anova.png', dpi=150, bbox_inches='tight')
plt.show()
print(" RQ3 plot saved as rq3_anova.png")


# SUMMARY TABLE
print_section("SUMMARY OF ALL HYPOTHESIS TESTS")

print(f"\n{'RQ':<6} {'Test':<22} {'t/F stat':>10} {'p-value':>12} {'Effect Size':>14} {'Decision':>15}")
print("-"*82)
print(f"{'RQ1':<6} {'One-tailed Welch t':<22} {t_stat:>10.4f} {p_one:>12.6f} {'d='+str(round(cohens_d,4)):>14} {'Reject H₀' if p_one<0.05 else 'Fail to Reject':>15}")
print(f"{'RQ2':<6} {'Two-tailed Welch t':<22} {t2:>10.4f} {p2:>12.6f} {'d='+str(round(cohens_d2,4)):>14} {'Reject H₀' if p2<0.05 else 'Fail to Reject':>15}")
print(f"{'RQ3':<6} {'One-way ANOVA':<22} {F_stat:>10.4f} {p_anova:>12.6f} {'η²='+str(round(eta_sq,4)):>14} {'Reject H₀' if p_anova<0.05 else 'Fail to Reject':>15}")

# Download all output plots
from google.colab import files
for fname in ['rq1_height.png', 'rq2_bmi.png', 'rq3_anova.png']:
    files.download(fname)
print("\n All plots downloaded!")

