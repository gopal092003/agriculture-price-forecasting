import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import seaborn as sns

PALETTE = sns.color_palette('tab10')
FMT = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

def volatility_plots(df):
    reports={}
    df['price_diff']    = df['avg_monthly_price'].diff()
    df['pct_change']    = df['avg_monthly_price'].pct_change() * 100
    df['rolling_std3']  = df['avg_monthly_price'].rolling(3).std()
    df['rolling_std6']  = df['avg_monthly_price'].rolling(6).std()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    bar_colors = ['crimson' if x < 0 else 'seagreen' for x in df['pct_change'].fillna(0)]
    axes[0].bar(df['date'], df['pct_change'], color=bar_colors, width=25, alpha=0.85)
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].axhline(df['pct_change'].mean(), color='navy', ls='--', lw=1,
                    label=f'Mean: {df["pct_change"].mean():.2f}%')
    axes[0].set_title('Month-over-Month % Price Change', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('% Change'); axes[0].legend()
    
    axes[1].plot(df['date'], df['rolling_std3'], color=PALETTE[1], lw=1.8, label='3-month Rolling Std')
    axes[1].plot(df['date'], df['rolling_std6'], color=PALETTE[2], lw=1.8, ls='--', label='6-month Rolling Std')
    axes[1].set_title('Rolling Std Dev (Price Volatility)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Std Dev'); axes[1].yaxis.set_major_formatter(FMT); axes[1].legend()
    
    axes[2].plot(df['date'], df['price_diff'].abs(), color=PALETTE[3], lw=1.5, label='|MoM Diff|')
    axes[2].set_title('Absolute MoM Price Change', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Date'); axes[2].set_ylabel('|dP|'); axes[2].yaxis.set_major_formatter(FMT); axes[2].legend()
    plt.tight_layout()

    yearly_vol = df.groupby('year')['avg_monthly_price'].std().round(0).fillna(0).astype(int)
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.bar(yearly_vol.index, yearly_vol.values,
           color=plt.cm.Reds(yearly_vol.values / yearly_vol.max()), edgecolor='white')
    ax.set_title('Price Volatility (Std Dev) by Year', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Std Dev'); ax.yaxis.set_major_formatter(FMT)
    plt.tight_layout(); plt.show()
    reports['Volatility'] = {
        'Most volatile year': f"{yearly_vol.idxmax()} (Std = {yearly_vol.max():,})",
        'Least volatile year': f"{yearly_vol.idxmin()} (Std = {yearly_vol.min():,})"
    }

    return(fig,fig2, reports['Volatility'],df)