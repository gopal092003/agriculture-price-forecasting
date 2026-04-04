import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FMT = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

def season_wise_data(df):
    SEASON_ORDER  = ['Winter','Summer','Monsoon','Post-Monsoon']
    SEASON_COLORS = {'Winter':'#4e9af1','Summer':'#f4a31c','Monsoon':'#2ecc71','Post-Monsoon':'#e74c3c'}
    
    season_yearly = df.groupby(['year','season'])['avg_monthly_price'].mean().reset_index()
    season_yearly['season'] = pd.Categorical(season_yearly['season'], categories=SEASON_ORDER, ordered=True)
    season_yearly.sort_values(['year','season'], inplace=True)
    return(season_yearly.to_string(index=False),season_yearly,SEASON_ORDER,SEASON_COLORS,df)

def season_wise_plots(df,season_yearly,SEASON_ORDER,SEASON_COLORS):
    reports={}
    fig1, ax = plt.subplots(figsize=(14, 5))
    for s in SEASON_ORDER:
        sub = season_yearly[season_yearly['season']==s]
        ax.plot(sub['year'], sub['avg_monthly_price'], marker='o', ms=5, lw=2,
                label=s, color=SEASON_COLORS[s])
    ax.set_title('Season-wise Price Trend Across Years', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Price'); ax.yaxis.set_major_formatter(FMT)
    ax.legend(title='Season', framealpha=0.8)
    plt.tight_layout()

    season_agg = df.groupby('season')['avg_monthly_price'].agg(['mean','std']).reindex(SEASON_ORDER)
    fig2, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(season_agg.index, season_agg['mean'],
                  color=[SEASON_COLORS[s] for s in season_agg.index],
                  yerr=season_agg['std'], capsize=6)
    for bar, val in zip(bars, season_agg['mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Average Price by Season (with Std Dev)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price'); ax.yaxis.set_major_formatter(FMT)
    plt.tight_layout(); plt.show()
    
    season_cv = df.groupby('season')['avg_monthly_price'].std() /             df.groupby('season')['avg_monthly_price'].mean() * 100
    reports['Season-wise CV%'] = season_cv.round(1).to_string()

    return (fig1,fig2, reports['Season-wise CV%'], df)