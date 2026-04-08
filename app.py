import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="FIFA Wage Amplification",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d0f; color: #e8e6df; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }
.dash-title { font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500;
              color: #e8e6df; letter-spacing: 0.08em; text-transform: uppercase; margin: 0; }
.dash-sub { font-size: 0.75rem; color: #555; margin-top: 3px;
            font-family: 'DM Mono', monospace; letter-spacing: 0.04em; }
.stat-card { background: #16161a; border: 1px solid #2a2a2e; border-radius: 8px; padding: 14px 16px; }
.stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em;
              font-family: 'DM Mono', monospace; margin-bottom: 5px; }
.stat-value { font-size: 1.7rem; font-weight: 600; font-family: 'DM Mono', monospace; line-height: 1; }
.stat-sub2 { font-size: 0.68rem; color: #444; margin-top: 4px; font-family: 'DM Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #2a2a2e; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; color: #555;
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 10px 18px; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: #e8e6df !important;
    border-bottom: 2px solid #4a9eff !important; background: transparent !important; }
.insight-box { background: #16161a; border-left: 3px solid #4a9eff; border-radius: 0 6px 6px 0;
               padding: 11px 15px; font-size: 0.8rem; color: #888; line-height: 1.7; margin-top: 10px; }
.insight-box b { color: #e8e6df; }
.insight-pink { border-left-color: #f472b6 !important; }
.insight-green { border-left-color: #3ecf8e !important; }
.result-box { background: #16161a; border: 1px solid #2a2a2e; border-radius: 8px;
              padding: 1.2rem 1.5rem; margin-top: 1rem; }
section[data-testid="stSidebar"] { background: #0d0d0f; border-right: 1px solid #2a2a2e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NAME_MAP = {
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom",
    "Republic of Ireland": "Ireland", "Korea Republic": "South Korea",
    "Korea DPR": "North Korea", "China PR": "China",
    "Chinese Taipei": "Taiwan", "Congo DR": "Democratic Republic of Congo",
    "Congo": "Republic of the Congo", "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde", "Guinea Bissau": "Guinea-Bissau",
    "Brunei Darussalam": "Brunei", "Swaziland": "Eswatini",
    "Puerto Rico": "United States", "Aruba": "Netherlands",
    "Curacao": "Netherlands", "Bermuda": "United Kingdom",
    "Montserrat": "United Kingdom", "Guam": "United States",
    "New Caledonia": "France", "Liechtenstein": "Switzerland",
}
MALE_LC = {
    'Premier League': 'United Kingdom', 'Championship': 'United Kingdom',
    'League One': 'United Kingdom', 'League Two': 'United Kingdom',
    'La Liga': 'Spain', 'La Liga 2': 'Spain',
    'Bundesliga': 'Germany', '2. Bundesliga': 'Germany', '3. Liga': 'Germany',
    'Ligue 1': 'France', 'Ligue 2': 'France', 'Serie A': 'Italy', 'Serie B': 'Italy',
    'Eredivisie': 'Netherlands', 'Liga Portugal': 'Portugal',
    'Major League Soccer': 'United States', 'K League 1': 'South Korea',
    'A-League': 'Australia', 'Super Lig': 'Turkey',
    'Jupiler Pro League': 'Belgium', 'Ekstraklasa': 'Poland',
    'Allsvenskan': 'Sweden', 'Eliteserien': 'Norway',
    'Super League': 'China', 'Pro League': 'Saudi Arabia',
    'Superliga': 'Denmark', 'Liga MX': 'Mexico',
    'Primera Division': 'Argentina', 'Liga Profesional': 'Argentina',
    'Liga BetPlay': 'Colombia',
}
FEMALE_LC = {'Feminine Division 1': 'France', "Women's Super League": 'United Kingdom'}
BG, GRID, MUTED = "#0d0d0f", "#1e1e22", "#777"
BLUE, AMBER, GREEN, RED, PINK = "#4a9eff", "#f0a500", "#3ecf8e", "#ff6b6b", "#f472b6"
FONT = "DM Mono, monospace"


def blayout(fig, height=400, margin=None):
    m = margin or dict(l=10, r=10, t=30, b=10)
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family=FONT, color=MUTED, size=11), margin=m,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10))
    return fig


def scard(col, label, val, color, sub=""):
    with col:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value" style="color:{color}">{val}</div>
          <div class="stat-sub2">{sub}</div>
        </div>""", unsafe_allow_html=True)


def prep(csv_path, lc_map):
    df   = pd.read_csv(csv_path)
    wage = pd.read_csv('global_minimum_wage.csv')
    df['country_mapped'] = df['nationality_name'].map(lambda x: NAME_MAP.get(x, x))
    mwc  = next(c for c in wage.columns if 'usd' in c.lower() and 'month' in c.lower())
    wc   = wage.dropna(subset=[mwc])
    merged = df.merge(
        wc[['Country', mwc]].rename(columns={mwc: 'mw_mo'}),
        left_on='country_mapped', right_on='Country', how='inner'
    )
    merged['mw_mo']           = pd.to_numeric(merged['mw_mo'], errors='coerce')
    merged['annual_minwage']  = merged['mw_mo'] * 12
    merged['annual_wage_usd'] = merged['wage_eur'] * 52 * 1.08
    merged['monthly_usd']     = merged['wage_eur'] * 52 / 12 * 1.08
    merged['amp_ratio']       = merged['annual_wage_usd'] / (merged['annual_minwage'] + 1)
    merged['amplification']   = merged['monthly_usd'] / (merged['mw_mo'] + 0.01)
    merged['log_amp']         = np.log1p(merged['amp_ratio'])
    merged['rating_bin']      = pd.cut(merged['overall'],
        bins=[39,50,60,70,80,90,95],
        labels=["40–50","51–60","61–70","71–80","81–90","91–95"])
    merged['league_country']  = merged['league_name'].map(lc_map)
    merged['is_migrant']      = merged['league_country'] != merged['country_mapped']
    return merged


@st.cache_data
def load_male():   return prep('fifa_players.csv',   MALE_LC)
@st.cache_data
def load_female(): return prep('female_players.csv', FEMALE_LC)


@st.cache_resource
def train_all(m_hash, f_hash):
    out = {}
    for key, df in [('male', load_male()), ('female', load_female())]:
        cols  = ['overall','potential','age','value_eur','league_name','league_level','wage_eur']
        avail = [c for c in cols if c in df.columns]
        feat  = [c for c in avail if c != 'wage_eur']
        data  = df[avail].dropna()
        if len(data) < 30:
            out[key] = None; continue
        X  = data[feat].copy()
        y  = np.log1p(data['wage_eur'])
        le = LabelEncoder()
        if 'league_name' in X.columns: X['league_name'] = le.fit_transform(X['league_name'])
        if 'value_eur'   in X.columns: X['value_eur']   = np.log1p(X['value_eur'])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        rf  = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=10, learning_rate=0.05, random_state=42)
        rf.fit(Xtr, ytr); hgb.fit(Xtr, ytr)
        rfp = np.expm1(rf.predict(Xte)); hgbp = np.expm1(hgb.predict(Xte)); ya = np.expm1(yte)
        out[key] = dict(
            rf=rf, hgb=hgb, le=le, features=feat,
            rf_mae=mean_absolute_error(ya,rfp),   rf_rmse=np.sqrt(mean_squared_error(ya,rfp)),   rf_r2=r2_score(ya,rfp),
            hgb_mae=mean_absolute_error(ya,hgbp), hgb_rmse=np.sqrt(mean_squared_error(ya,hgbp)), hgb_r2=r2_score(ya,hgbp),
            rf_feat_imp=list(zip(feat, rf.feature_importances_)),
            y_actual=ya.values[:500], rf_preds=rfp[:500], hgb_preds=hgbp[:500],
        )
    return out


def nat_stats(df, mn=20):
    rows = []
    for nat in df['nationality_name'].value_counts().head(20).index:
        s = df[df['nationality_name']==nat].dropna(subset=['amp_ratio'])
        if len(s) < mn: continue
        r, p = stats.spearmanr(s['overall'], s['amp_ratio'])
        mw = s['annual_minwage'].dropna()
        rows.append(dict(nationality=nat, n=len(s), spearman_rho=round(r,3),
                         median_amp=round(s['amp_ratio'].median(),1),
                         annual_minwage=round(mw.iloc[0] if len(mw) else 0, 0)))
    return pd.DataFrame(rows).sort_values('spearman_rho', ascending=False)


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models — ~30 seconds first run..."):
    male_df   = load_male()
    female_df = load_female()
    models    = train_all(len(male_df), len(female_df))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚥ Dataset")
    mode = st.radio("View", ["Male players", "Female players"])
    st.divider()
    st.markdown("### 🎛️ Filters")
    st.caption("All charts update instantly")
    df_full = male_df if mode == "Male players" else female_df
    accent  = BLUE    if mode == "Male players" else PINK
    gender  = "male"  if mode == "Male players" else "female"
    icon_g  = "♂"     if mode == "Male players" else "♀"
    mdl     = models.get('male') if mode == "Male players" else models.get('female')

    rating_range = st.slider("Overall rating", 40, 95, (40, 95))
    age_range    = st.slider("Age", int(df_full['age'].min()), int(df_full['age'].max()),
                              (int(df_full['age'].min()), int(df_full['age'].max())))
    sel_nats    = st.multiselect("Nationalities", sorted(df_full['nationality_name'].dropna().unique()), placeholder="All")
    sel_leagues = st.multiselect("Leagues",       sorted(df_full['league_name'].dropna().unique()),      placeholder="All")
    st.divider()
    st.markdown("### ⚙️ Display")
    top_n     = st.slider("Top N (charts)", 5, 25, 15)
    log_scale = st.toggle("Log scale on amp axis", value=True)
    st.divider()
    st.caption("wage_eur = weekly · EUR→USD @ 1.08")

# ── Filter ────────────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[(df['overall'] >= rating_range[0]) & (df['overall'] <= rating_range[1])]
df = df[(df['age']     >= age_range[0])    & (df['age']     <= age_range[1])]
if sel_nats:    df = df[df['nationality_name'].isin(sel_nats)]
if sel_leagues: df = df[df['league_name'].isin(sel_leagues)]
clean = df.dropna(subset=['amp_ratio','overall']).copy()

if len(clean) < 10:
    st.warning("Too few players after filtering — relax the filters.")
    st.stop()

# ── Core stats ────────────────────────────────────────────────────────────────
rho,   _ = stats.spearmanr(clean['overall'], clean['amp_ratio'])
r_log, _ = stats.pearsonr(clean['overall'], clean['log_amp'].fillna(clean['log_amp'].median()))
bin_s = (
    clean.groupby('rating_bin', observed=True)['amp_ratio']
    .agg(median='median', p25=lambda x: x.quantile(0.25),
         p75=lambda x: x.quantile(0.75), count='count')
    .round(1).reset_index()
)
nat_df   = nat_stats(clean, mn=10 if gender=='female' else 20)
med_e    = bin_s[bin_s['rating_bin']=='91–95']['median'].values
med_l    = bin_s[bin_s['rating_bin']=='40–50']['median'].values
med_e    = med_e[0] if len(med_e) else 0
med_l    = med_l[0] if len(med_l) else 1

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""<div style="border-bottom:1px solid #2a2a2e;padding-bottom:1rem;margin-bottom:1.2rem">
  <p class="dash-title">⚽ FIFA Wage Amplification Dashboard
    <span style="color:{accent};font-size:0.85rem">{icon_g} {gender.upper()}</span></p>
  <p class="dash-sub">player annual wage ÷ home country annual minimum wage
    · {len(clean):,} players · {clean['country_mapped'].nunique()} countries
    · all charts respond to sidebar filters</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
scard(c1,"Spearman ρ",     f"{rho:.3f}",      accent,    "skill → amp ratio")
scard(c2,"R² (log scale)", f"{r_log**2:.1%}", AMBER,     "variance explained")
scard(c3,"Players",        f"{len(clean):,}", "#e8e6df", "after filters")
scard(c4,"Median 91–95",   f"{med_e:,.0f}×",  GREEN,     "vs min wage worker")
scard(c5,"Median 40–50",   f"{med_l:,.0f}×",  RED,       "vs min wage worker")
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "MIGRATION vs AMPLIFICATION","SKILL vs AMPLIFICATION",
    "BY NATIONALITY","GLOBALISATION","PLAYER EXPLORER","WAGE PREDICTOR","COUNTRY DEEP DIVE",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    mig = (clean.groupby('nationality_name')
           .agg(count=('overall','size'), median_amp=('amp_ratio','median'))
           .reset_index().sort_values('count', ascending=False).head(top_n))
    sort_by = st.radio("Sort bars by",["Player count","Amp ratio"],horizontal=True,key="t1s")
    mig = mig.sort_values("count" if sort_by=="Player count" else "median_amp", ascending=True)
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(y=mig['nationality_name'],x=mig['count'],name="Player count",
        orientation="h",marker_color=accent,marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Players: %{x:,}<extra></extra>"),secondary_y=False)
    fig.add_trace(go.Scatter(y=mig['nationality_name'],x=mig['median_amp'],name="Median amp ratio",
        mode="lines+markers",line=dict(color=RED,width=2),marker=dict(size=8,color=RED),
        hovertemplate="<b>%{y}</b><br>Amp ratio: %{x:.1f}×<extra></extra>"),secondary_y=True)
    blayout(fig,height=max(380,top_n*28))
    fig.update_layout(legend=dict(orientation="h",y=1.06))
    fig.update_xaxes(title_text="Number of players",title_font=dict(color=accent,size=11))
    fig.update_yaxes(title_text="Median amp ratio (×)",title_font=dict(color=RED,size=11),secondary_y=True)
    st.plotly_chart(fig,use_container_width=True)
    if gender=='male':
        st.markdown("""<div class="insight-box"><b>Brazil</b> sends large numbers abroad at high amplification.
          <b>UK</b> exports the most players but lowest ratio due to high home minimum wage.
          Volume and economic amplification are largely independent of each other.</div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div class="insight-box insight-pink"><b>Canada</b> is the top exporter by volume.
          <b>Haiti</b> has the highest amplification (~98.5×). The female game is significantly
          less globalised — only 2 leagues in this dataset.</div>""",unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    col_l,col_r = st.columns([2,1])
    with col_l:
        ct = st.radio("View",["Bar (median per tier)","Scatter (individual players)"],horizontal=True,key="t2c")
        if ct == "Bar (median per tier)":
            bc = [accent]*4+[AMBER,GREEN]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=bin_s['rating_bin'].astype(str),
                y=bin_s['p75']-bin_s['p25'],base=bin_s['p25'],name="IQR",
                marker_color="rgba(74,158,255,0.10)",marker_line_width=0,hoverinfo="skip"))
            fig2.add_trace(go.Bar(x=bin_s['rating_bin'].astype(str),y=bin_s['median'],
                name="Median",marker_color=bc,marker_line_width=0,width=0.4,
                text=bin_s['median'].apply(lambda v:f"{v:,.0f}×"),textposition="outside",
                textfont=dict(size=10,color=MUTED),
                hovertemplate="<b>%{x}</b><br>Median: %{y:,.0f}×<br>n=%{customdata:,}<extra></extra>",
                customdata=bin_s['count']))
            blayout(fig2,height=400)
            fig2.update_layout(barmode="overlay",showlegend=False,
                yaxis_type="log" if log_scale else "linear",
                xaxis=dict(title="Overall rating tier",gridcolor=GRID),
                yaxis=dict(title="Amplification ratio"+((" (log)") if log_scale else ""),
                           gridcolor=GRID,ticksuffix="×"))
        else:
            smp = clean.sample(min(5000,len(clean)),random_state=42)
            fig2 = px.scatter(smp,x='overall',y='amp_ratio',color='amp_ratio',
                color_continuous_scale=[[0,"#1a3a5c"],[0.5,accent],[1,GREEN]],opacity=0.35,
                hover_data={'nationality_name':True,'overall':True,'amp_ratio':':.1f','wage_eur':True},
                labels={'overall':'Overall rating','amp_ratio':'Amp ratio','nationality_name':'Nationality'})
            xr = np.linspace(clean['overall'].min(),clean['overall'].max(),200)
            sl,ic,*_ = stats.linregress(clean['overall'],clean['log_amp'].fillna(clean['log_amp'].median()))
            fig2.add_trace(go.Scatter(x=xr,y=np.exp(sl*xr+ic),mode="lines",
                line=dict(color=RED,width=2,dash="dash"),name=f"Trend (ρ={rho:.2f})",showlegend=True))
            blayout(fig2,height=420)
            fig2.update_layout(yaxis_type="log" if log_scale else "linear",coloraxis_showscale=False,
                yaxis=dict(title="Amp ratio"+((" (log)") if log_scale else ""),ticksuffix="×"),
                xaxis=dict(title="Overall rating"))
        st.plotly_chart(fig2,use_container_width=True)
    with col_r:
        st.markdown("<br>",unsafe_allow_html=True)
        for lbl,val,col,sub in [("Spearman ρ",f"{rho:.3f}",accent,"skill vs amp"),
                                  ("R² (log)",f"{r_log**2:.1%}",AMBER,"variance explained"),
                                  ("Players",f"{len(clean):,}","#e8e6df","in filtered set")]:
            st.markdown(f"""<div class="stat-card" style="margin-bottom:10px">
              <div class="stat-label">{lbl}</div><div class="stat-value" style="color:{col}">{val}</div>
              <div class="stat-sub2">{sub}</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        db = bin_s[['rating_bin','median','count']].copy()
        db.columns = ['Tier','Median ×','Count']
        db['Median ×'] = db['Median ×'].apply(lambda v:f"{v:,.1f}×")
        db['Count']    = db['Count'].apply(lambda v:f"{int(v):,}")
        st.dataframe(db,use_container_width=True,hide_index=True,height=240)
    ratio = int(med_e/med_l) if med_l>0 else "N/A"
    st.markdown(f"""<div class="insight-box">Spearman ρ = <b>{rho:.2f}</b>, R² = <b>{r_log**2:.0%}</b> on log scale.
      Median elite player earns <b>{med_e:,.0f}×</b> home min wage — roughly <b>{ratio}×</b> more than lowest tier.
      Switch to scatter to see individual distribution.</div>""",unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if len(nat_df)==0:
        st.info("Not enough players per nationality for this analysis with current filters.")
    else:
        cb = st.radio("Colour bars by",["Spearman ρ strength","Median amp ratio","Home min wage"],horizontal=True,key="t3c")
        xmap = {"Spearman ρ strength":("spearman_rho","Spearman ρ"),
                "Median amp ratio":("median_amp","Median amp ratio (×)"),
                "Home min wage":("annual_minwage","Annual min wage (USD)")}
        xc,xt = xmap[cb]
        ns = nat_df.sort_values(xc)
        if cb=="Spearman ρ strength":
            brc = [GREEN if r>=0.88 else accent if r>=0.70 else "#2a6496" for r in ns['spearman_rho']]
        elif cb=="Median amp ratio":
            brc = px.colors.sample_colorscale("Blues",np.linspace(0.3,1,len(ns)))
        else:
            brc = px.colors.sample_colorscale("Reds_r",np.linspace(0.2,0.9,len(ns)))
        fmt_fn = (lambda v:f"{v:.3f}") if xc=='spearman_rho' else \
                 (lambda v:f"{v:,.1f}×") if xc=='median_amp' else (lambda v:f"${v:,.0f}")
        cc,ct2 = st.columns([3,2])
        with cc:
            fig3 = go.Figure(go.Bar(x=ns[xc],y=ns['nationality'],orientation="h",
                marker_color=brc,marker_line_width=0,
                text=ns[xc].apply(fmt_fn),textposition="outside",textfont=dict(size=9,color=MUTED),
                hovertemplate="<b>%{y}</b><br>"+xt+": %{x}<br>n=%{customdata:,}<extra></extra>",
                customdata=ns['n']))
            if xc=='spearman_rho':
                fig3.add_vline(x=rho,line_dash="dot",line_color="#333",line_width=1,
                               annotation_text=f"overall ρ = {rho:.2f}",
                               annotation_font_size=9,annotation_font_color="#555")
            blayout(fig3,height=max(400,len(ns)*26),margin=dict(l=10,r=70,t=10,b=10))
            fig3.update_layout(showlegend=False,xaxis=dict(title=xt,gridcolor=GRID),yaxis=dict(showgrid=False))
            st.plotly_chart(fig3,use_container_width=True)
        with ct2:
            st.markdown("<br>",unsafe_allow_html=True)
            disp = nat_df[['nationality','n','spearman_rho','median_amp','annual_minwage']].copy()
            disp.columns = ['Country','N','ρ','Med ×','Min Wage/yr']
            disp['Min Wage/yr'] = disp['Min Wage/yr'].apply(lambda v:f"${v:,.0f}")
            disp['Med ×']       = disp['Med ×'].apply(lambda v:f"{v:,.1f}×")
            disp['N']           = disp['N'].apply(lambda v:f"{v:,}")
            st.dataframe(disp,use_container_width=True,height=max(400,len(disp)*35+40),hide_index=True)
        st.markdown("""<div class="insight-box">Countries with high Spearman ρ show tight skill-to-pay links.
          Countries with low ρ have more noise — often due to mixed league environments.
          Switch the colour mode to explore how min wage relates to amplification.</div>""",unsafe_allow_html=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    mig_rate = clean['is_migrant'].mean()
    mbc = (clean[clean['is_migrant']].groupby('nationality_name')
           .agg(migrant_count=('overall','count'),avg_amp=('amp_ratio','median'))
           .reset_index().sort_values('migrant_count',ascending=False))
    mbl = (clean[clean['is_migrant']].groupby('league_name')
           .agg(migrant_count=('overall','count'),
                unique_nats=('nationality_name','nunique'),
                avg_amp=('amp_ratio','median'))
           .reset_index().sort_values('migrant_count',ascending=False))
    gl = clean.groupby('league_name').agg(total=('overall','count'),migrants=('is_migrant','sum'),
                                           nats=('nationality_name','nunique')).reset_index()
    gl['mig_rate'] = gl['migrants']/gl['total']
    gl['g_index']  = gl['mig_rate']*0.5 + (gl['nats']/gl['nats'].max())*0.5
    mp = 50 if gender=='female' else 200
    gl = gl[gl['total']>=mp].sort_values('g_index',ascending=False)
    ts = mbc.iloc[0] if len(mbc) else None
    td = gl.iloc[0]  if len(gl)  else None
    g1,g2,g3,g4 = st.columns(4)
    scard(g1,"Overall migration rate",f"{mig_rate:.1%}",accent,"play outside home country")
    scard(g2,"Top sending nation",
          ts['nationality_name'] if ts is not None else "—",AMBER,
          f"{int(ts['migrant_count']):,} migrants" if ts is not None else "")
    scard(g3,"Most diverse league",
          td['league_name'] if td is not None else "—",GREEN,
          f"{int(td['nats'])} nationalities" if td is not None else "")
    scard(g4,"Leagues analysed",f"{len(gl):,}",RED,f"with ≥{mp} players")
    st.markdown("<div style='height:14px'></div>",unsafe_allow_html=True)
    t4v = st.radio("View",["Countries sending migrants","League globalisation index","Migrant amplification by league"],horizontal=True,key="t4v")
    if t4v=="Countries sending migrants":
        n4 = st.slider("Top N",5,max(5,len(mbc)),min(top_n,len(mbc)),key="t4n")
        s4 = st.radio("Sort by",["Migrant count","Amplification"],horizontal=True,key="t4s")
        ts4 = mbc.head(n4).sort_values("migrant_count" if s4=="Migrant count" else "avg_amp",ascending=True)
        fg = make_subplots(specs=[[{"secondary_y":True}]])
        fg.add_trace(go.Bar(y=ts4['nationality_name'],x=ts4['migrant_count'],name="Migrant count",
            orientation="h",marker_color=accent,marker_line_width=0,opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Migrants: %{x:,}<extra></extra>"),secondary_y=False)
        fg.add_trace(go.Scatter(y=ts4['nationality_name'],x=ts4['avg_amp'],name="Median amp ratio",
            mode="lines+markers",line=dict(color=AMBER,width=2),marker=dict(size=8,color=AMBER),
            hovertemplate="<b>%{y}</b><br>Amp: %{x:.1f}×<extra></extra>"),secondary_y=True)
        blayout(fg,height=max(380,n4*28))
        fg.update_layout(legend=dict(orientation="h",y=1.06))
        fg.update_xaxes(title_text="Migrant players",title_font=dict(color=accent,size=11))
        fg.update_yaxes(title_text="Median amp ratio (×)",title_font=dict(color=AMBER,size=11),secondary_y=True)
        st.plotly_chart(fg,use_container_width=True)
    elif t4v=="League globalisation index":
        n4l = st.slider("Top N leagues",5,max(5,len(gl)),min(15,len(gl)),key="t4nl")
        glt = gl.head(n4l).sort_values('g_index',ascending=True)
        glc = px.colors.sample_colorscale("Blues",np.linspace(0.3,1,len(glt)))
        fgl = go.Figure(go.Bar(y=glt['league_name'],x=glt['g_index'],orientation="h",
            marker_color=glc,marker_line_width=0,
            text=glt['g_index'].apply(lambda v:f"{v:.3f}"),textposition="outside",textfont=dict(size=9,color=MUTED),
            customdata=np.stack([glt['mig_rate'],glt['nats']],axis=-1),
            hovertemplate="<b>%{y}</b><br>G-Index: %{x:.3f}<br>Mig: %{customdata[0]:.1%}<br>Nations: %{customdata[1]}<extra></extra>"))
        blayout(fgl,height=max(380,n4l*28),margin=dict(l=10,r=60,t=10,b=10))
        fgl.update_layout(showlegend=False,xaxis=dict(title="Globalisation index",gridcolor=GRID),yaxis=dict(showgrid=False))
        st.plotly_chart(fgl,use_container_width=True)
        dgl = gl.head(n4l)[['league_name','total','migrants','mig_rate','nats','g_index']].copy()
        dgl.columns = ['League','Total','Migrants','Mig. Rate','Nations','G-Index']
        dgl['Mig. Rate'] = dgl['Mig. Rate'].apply(lambda v:f"{v:.1%}")
        dgl['G-Index']   = dgl['G-Index'].apply(lambda v:f"{v:.3f}")
        dgl['Total']     = dgl['Total'].apply(lambda v:f"{int(v):,}")
        dgl['Migrants']  = dgl['Migrants'].apply(lambda v:f"{int(v):,}")
        st.dataframe(dgl,use_container_width=True,hide_index=True,height=320)
    else:
        n4m = st.slider("Top N leagues",5,max(5,len(mbl)),min(15,len(mbl)),key="t4nm")
        mb  = mbl.head(n4m).sort_values('avg_amp',ascending=True)
        mc  = [GREEN if v>=mb['avg_amp'].quantile(0.75) else accent if v>=mb['avg_amp'].median() else "#2a6496" for v in mb['avg_amp']]
        fmb = go.Figure(go.Bar(y=mb['league_name'],x=mb['avg_amp'],orientation="h",
            marker_color=mc,marker_line_width=0,
            text=mb['avg_amp'].apply(lambda v:f"{v:,.1f}×"),textposition="outside",textfont=dict(size=9,color=MUTED),
            customdata=np.stack([mb['migrant_count'],mb['unique_nats']],axis=-1),
            hovertemplate="<b>%{y}</b><br>Amp: %{x:.1f}×<br>Migrants: %{customdata[0]:,}<br>Nations: %{customdata[1]}<extra></extra>"))
        blayout(fmb,height=max(380,n4m*28),margin=dict(l=10,r=70,t=10,b=10))
        fmb.update_layout(showlegend=False,xaxis=dict(title="Median amp ratio (×)",gridcolor=GRID,ticksuffix="×"),yaxis=dict(showgrid=False))
        st.plotly_chart(fmb,use_container_width=True)
    st.markdown(f"""<div class="insight-box"><b>{mig_rate:.1%}</b> of players compete outside their home country.
      High-amplification migrants come from low min-wage nations playing in top leagues.
      The <b>Globalisation Index</b> combines migration rate and nationality diversity equally.</div>""",unsafe_allow_html=True)

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("##### Drill into individual players")
    e1,e2,e3 = st.columns(3)
    with e1: nat_f = st.selectbox("Nationality",["All"]+sorted(clean['nationality_name'].dropna().unique().tolist()),key="exp_nat")
    with e2: mr = st.slider("Min overall rating",40,95,60,key="exp_r")
    with e3: sc = st.selectbox("Sort by",["amp_ratio","overall","wage_eur","annual_wage_usd"],key="exp_s")
    exp = clean[clean['overall']>=mr].copy()
    if nat_f!="All": exp = exp[exp['nationality_name']==nat_f]
    exp = exp.sort_values(sc,ascending=False).head(300)
    ecols = ['nationality_name','league_name','overall','age','wage_eur','annual_wage_usd','annual_minwage','amp_ratio']
    ecols = [c for c in ecols if c in exp.columns]
    de = exp[ecols].copy()
    de.columns = [c.replace('_',' ').title() for c in ecols]
    for c,fmt in [('Wage Eur','€{:,.0f}'),('Annual Wage Usd','${:,.0f}'),('Annual Minwage','${:,.0f}')]:
        if c in de.columns: de[c] = de[c].apply(lambda v:fmt.format(v) if pd.notna(v) else "—")
    if 'Amp Ratio' in de.columns: de['Amp Ratio'] = de['Amp Ratio'].apply(lambda v:f"{v:,.1f}×" if pd.notna(v) else "—")
    st.dataframe(de,use_container_width=True,height=380,hide_index=True)
    st.caption(f"Top 300 of {len(exp):,} players · sorted by {sc}")
    if len(exp)>10:
        sp = exp.sample(min(3000,len(exp)),random_state=1)
        fe = px.scatter(sp,x='overall',y='amp_ratio',
            color='nationality_name' if nat_f=="All" else 'overall',opacity=0.45,
            color_continuous_scale="Blues" if nat_f!="All" else None,
            hover_data={'nationality_name':True,'overall':True,'amp_ratio':':.1f','wage_eur':True},
            labels={'overall':'Overall rating','amp_ratio':'Amp ratio','nationality_name':'Nationality'},
            title=f"{'All nationalities' if nat_f=='All' else nat_f} · rating ≥ {mr}")
        blayout(fe,height=340)
        fe.update_layout(yaxis_type="log" if log_scale else "linear",yaxis=dict(ticksuffix="×"),
            showlegend=nat_f=="All",legend=dict(font=dict(size=9),itemsizing="constant"))
        st.plotly_chart(fe,use_container_width=True)

# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("##### Predict a player's weekly wage")
    if mdl is None:
        st.warning("Model not available.")
    else:
        st.markdown("---")
        st.markdown("#### Search for a nationality profile")
        srch = st.text_input("Search by nationality",placeholder="e.g. Brazil, France, Canada...",key="srch")
        if srch:
            res = df_full[df_full['nationality_name'].str.contains(srch,case=False,na=False)]
            if len(res)>0:
                dr = res[['nationality_name','league_name','overall','age','wage_eur','amp_ratio']].copy()
                dr.columns = ['Nationality','League','Rating','Age','Wage (€/wk)','Amp Ratio']
                dr['Amp Ratio'] = dr['Amp Ratio'].apply(lambda v:f"{v:.1f}×" if pd.notna(v) else "—")
                st.dataframe(dr.sort_values('Rating',ascending=False).reset_index(drop=True),
                             use_container_width=True,height=250,hide_index=True)
            else: st.warning("No players found.")
        st.markdown("---")
        st.markdown("#### Build a player profile")
        w1,w2 = st.columns(2)
        with w1:
            wov = st.slider("Overall rating",40,95,80,key="wov")
            wpt = st.slider("Potential",40,95,85,key="wpt")
            wag = st.slider("Age",16,40,24,key="wag")
        with w2:
            dv   = 500_000 if gender=='female' else 30_000_000
            wval = st.number_input("Market value (€)",min_value=0,value=dv,step=50_000,key="wval")
            wlg  = st.selectbox("League",sorted(df_full['league_name'].dropna().unique()),key="wlg")
            wlv  = st.selectbox("League level",[1,2,3,4,5],key="wlv")
            wnat = st.selectbox("Nationality",sorted(df_full['country_mapped'].dropna().unique()),key="wnat")
        if st.button("Predict wage",key="wpbtn",use_container_width=True):
            if wlg not in mdl['le'].classes_:
                st.error("League not in model — try a different league.")
            else:
                iv = []
                for f in mdl['features']:
                    if f=='overall':       iv.append(wov)
                    elif f=='potential':   iv.append(wpt)
                    elif f=='age':         iv.append(wag)
                    elif f=='value_eur':   iv.append(np.log1p(wval))
                    elif f=='league_name': iv.append(mdl['le'].transform([wlg])[0])
                    elif f=='league_level':iv.append(wlv)
                    else:                  iv.append(0)
                idf  = pd.DataFrame([iv],columns=mdl['features'])
                rfw  = np.expm1(mdl['rf'].predict(idf)[0])
                hgbw = np.expm1(mdl['hgb'].predict(idf)[0])
                avgw = (rfw+hgbw)/2
                avgm = avgw*52/12
                nmw  = df_full[df_full['country_mapped']==wnat]['mw_mo'].dropna().values
                amp  = (avgm*1.08)/nmw[0] if len(nmw)>0 else None
                st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
                p1,p2,p3,p4 = st.columns(4)
                scard(p1,"RF predicted weekly", f"€{rfw:,.0f}",  BLUE,  "Random Forest")
                scard(p2,"HGB predicted weekly",f"€{hgbw:,.0f}", GREEN, "Gradient Boosting")
                scard(p3,"Average weekly",       f"€{avgw:,.0f}", accent,"RF + HGB average")
                if amp: scard(p4,"Amplification ratio",f"{amp:.1f}×",AMBER,f"vs {wnat} min wage")
                st.markdown(f"""<div class="result-box">
                  <div style="font-size:0.7rem;color:#555;font-family:'DM Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">Prediction summary</div>
                  <div style="font-size:2rem;font-weight:700;font-family:'DM Mono',monospace;color:{accent}">
                    €{avgw:,.0f} / week</div>
                  <p style="color:#666;margin-top:8px;font-size:0.9rem;line-height:1.6">
                    A <b style="color:#e8e6df">{wnat}</b> {gender} player rated
                    <b style="color:#e8e6df">{wov}</b> in <b style="color:#e8e6df">{wlg}</b>.
                    {"Earns <b style='color:"+AMBER+"'>"+f"{amp:.0f}×</b> home min wage." if amp else ""}
                    RF: <b style="color:{BLUE}">€{rfw:,.0f}</b>/wk ·
                    HGB: <b style="color:{GREEN}">€{hgbw:,.0f}</b>/wk
                  </p></div>""",unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Model performance")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        scard(m1,"RF MAE",  f"€{mdl['rf_mae']:,.0f}",  BLUE, "Random Forest")
        scard(m2,"RF RMSE", f"€{mdl['rf_rmse']:,.0f}", BLUE, "Random Forest")
        scard(m3,"RF R²",   f"{mdl['rf_r2']:.4f}",     BLUE, "Random Forest")
        scard(m4,"HGB MAE", f"€{mdl['hgb_mae']:,.0f}", GREEN,"Gradient Boosting")
        scard(m5,"HGB RMSE",f"€{mdl['hgb_rmse']:,.0f}",GREEN,"Gradient Boosting")
        scard(m6,"HGB R²",  f"{mdl['hgb_r2']:.4f}",   GREEN,"Gradient Boosting")
        st.markdown("<div style='height:14px'></div>",unsafe_allow_html=True)
        mc1,mc2 = st.columns(2)
        with mc1:
            fi = sorted(mdl['rf_feat_imp'],key=lambda x:x[1])
            fn,fi2 = zip(*fi)
            ffi = go.Figure(go.Bar(y=list(fn),x=list(fi2),orientation="h",
                marker_color=accent,marker_line_width=0,
                text=[f"{v*100:.1f}%" for v in fi2],textposition="outside",textfont=dict(size=9,color=MUTED)))
            blayout(ffi,height=300,margin=dict(l=10,r=60,t=10,b=10))
            ffi.update_layout(showlegend=False,xaxis=dict(title="Feature importance",gridcolor=GRID),yaxis=dict(showgrid=False))
            st.plotly_chart(ffi,use_container_width=True)
        with mc2:
            ya = mdl['y_actual']; rfp = mdl['rf_preds']; hgbp = mdl['hgb_preds']
            mv = max(float(ya.max()),float(rfp.max()))
            fav = go.Figure()
            fav.add_trace(go.Scatter(x=ya,y=rfp,mode="markers",
                marker=dict(color=BLUE,size=4,opacity=0.4),name="Random Forest",
                hovertemplate="Actual: €%{x:,.0f}<br>RF: €%{y:,.0f}<extra></extra>"))
            fav.add_trace(go.Scatter(x=ya,y=hgbp,mode="markers",
                marker=dict(color=GREEN,size=4,opacity=0.4),name="Gradient Boosting",
                hovertemplate="Actual: €%{x:,.0f}<br>HGB: €%{y:,.0f}<extra></extra>"))
            fav.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode="lines",
                line=dict(color=RED,width=1.5,dash="dash"),name="Perfect fit"))
            blayout(fav,height=300)
            fav.update_layout(xaxis=dict(title="Actual wage (€)",gridcolor=GRID),
                yaxis=dict(title="Predicted wage (€)",gridcolor=GRID),legend=dict(orientation="h",y=1.05))
            st.plotly_chart(fav,use_container_width=True)

# ── TAB 7 — COUNTRY DEEP DIVE ─────────────────────────────────────────────────
with tab7:
    st.markdown("##### Country deep dive")
    st.caption("Search any nationality in the dataset for a full breakdown of amplification stats, league distribution, and top/bottom players.")

    all_countries = sorted(df_full['nationality_name'].dropna().unique().tolist())
    selected_country = st.selectbox("Select a country", all_countries, key="cdd_country")

    country_df = clean[clean['nationality_name'].str.lower() == selected_country.lower()].copy()

    if len(country_df) == 0:
        st.warning(f"No players found for '{selected_country}' in the current filtered dataset. Try adjusting sidebar filters.")
    else:
        mw_monthly = country_df['mw_mo'].iloc[0] if 'mw_mo' in country_df.columns else 0
        mw_weekly  = (mw_monthly * 12 / 52) / 1.08 if mw_monthly else 0

        # ── Summary stats ─────────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        s1,s2,s3,s4,s5 = st.columns(5)
        scard(s1, "Total players",    f"{len(country_df):,}",                 accent,  selected_country)
        scard(s2, "Min wage (USD/mo)",f"${mw_monthly:,.2f}",                  AMBER,   "home country")
        scard(s3, "Mean amp ratio",   f"{country_df['amp_ratio'].mean():.1f}×", GREEN,  "all players")
        scard(s4, "Median amp ratio", f"{country_df['amp_ratio'].median():.1f}×", BLUE, "all players")
        scard(s5, "Max amp ratio",    f"{country_df['amp_ratio'].max():.0f}×",  RED,    "top player")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # ── Amp distribution ──────────────────────────────────────────────────
        with col1:
            st.markdown("**Amplification distribution**")
            clip_v = country_df['amp_ratio'].quantile(0.95)
            fig_cd1 = go.Figure(go.Histogram(
                x=country_df['amp_ratio'].clip(upper=clip_v),
                nbinsx=30, marker_color=accent, marker_line_width=0, opacity=0.85,
                hovertemplate="Amp ratio: %{x:.1f}×<br>Count: %{y}<extra></extra>",
            ))
            blayout(fig_cd1, height=280)
            fig_cd1.update_layout(showlegend=False,
                xaxis=dict(title="Amp ratio (clipped at 95th pct)", gridcolor=GRID, ticksuffix="×"),
                yaxis=dict(title="Players", gridcolor=GRID))
            st.plotly_chart(fig_cd1, use_container_width=True)

        # ── Amp by league ─────────────────────────────────────────────────────
        with col2:
            st.markdown("**Amplification by league**")
            lg_stats = (country_df.groupby('league_name')['amp_ratio']
                        .agg(mean='mean', count='count')
                        .round(2).sort_values('mean', ascending=True).head(10))
            lc = px.colors.sample_colorscale("Blues", np.linspace(0.3, 1, len(lg_stats)))
            fig_cd2 = go.Figure(go.Bar(
                y=lg_stats.index, x=lg_stats['mean'],
                orientation="h", marker_color=lc, marker_line_width=0,
                text=lg_stats['mean'].apply(lambda v: f"{v:.1f}×"),
                textposition="outside", textfont=dict(size=9, color=MUTED),
                customdata=lg_stats['count'],
                hovertemplate="<b>%{y}</b><br>Mean amp: %{x:.1f}×<br>Players: %{customdata}<extra></extra>",
            ))
            blayout(fig_cd2, height=280, margin=dict(l=10, r=60, t=10, b=10))
            fig_cd2.update_layout(showlegend=False,
                xaxis=dict(title="Mean amp ratio (×)", gridcolor=GRID, ticksuffix="×"),
                yaxis=dict(showgrid=False))
            st.plotly_chart(fig_cd2, use_container_width=True)

        # ── Skill vs amp scatter ───────────────────────────────────────────────
        st.markdown("**Skill rating vs amplification**")
        fig_cd3 = px.scatter(
            country_df.sample(min(2000, len(country_df)), random_state=42),
            x='overall', y='amp_ratio', color='amp_ratio',
            color_continuous_scale=[[0,"#1a3a5c"],[0.5,accent],[1,GREEN]],
            opacity=0.5,
            hover_data={'league_name': True, 'overall': True, 'amp_ratio': ':.1f', 'wage_eur': True},
            labels={'overall':'Overall rating','amp_ratio':'Amp ratio','league_name':'League'},
        )
        if len(country_df) > 5:
            xr = np.linspace(country_df['overall'].min(), country_df['overall'].max(), 200)
            sl, ic, *_ = stats.linregress(country_df['overall'],
                                           country_df['log_amp'].fillna(country_df['log_amp'].median()))
            fig_cd3.add_trace(go.Scatter(x=xr, y=np.exp(sl*xr+ic), mode="lines",
                line=dict(color=RED, width=2, dash="dash"), name="Trend", showlegend=True))
        blayout(fig_cd3, height=320)
        fig_cd3.update_layout(
            yaxis_type="log" if log_scale else "linear",
            coloraxis_showscale=False,
            yaxis=dict(title="Amp ratio"+((" (log)") if log_scale else ""), ticksuffix="×"),
            xaxis=dict(title="Overall rating"))
        st.plotly_chart(fig_cd3, use_container_width=True)

        # ── Top & bottom 5 ────────────────────────────────────────────────────
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Top 5 players by amplification**")
            top5_cols = ['overall','league_name','wage_eur','amp_ratio']
            top5_cols = [c for c in top5_cols if c in country_df.columns]
            top5 = country_df.nlargest(5,'amp_ratio')[top5_cols].copy()
            top5.columns = [c.replace('_',' ').title() for c in top5_cols]
            if 'Wage Eur' in top5.columns:
                top5['Wage Eur'] = top5['Wage Eur'].apply(lambda v: f"€{v:,.0f}")
            if 'Amp Ratio' in top5.columns:
                top5['Amp Ratio'] = top5['Amp Ratio'].apply(lambda v: f"{v:,.1f}×")
            st.dataframe(top5, use_container_width=True, hide_index=True, height=220)

        with col4:
            st.markdown("**Bottom 5 players by amplification**")
            bot5 = country_df.nsmallest(5,'amp_ratio')[top5_cols].copy()
            bot5.columns = [c.replace('_',' ').title() for c in top5_cols]
            if 'Wage Eur' in bot5.columns:
                bot5['Wage Eur'] = bot5['Wage Eur'].apply(lambda v: f"€{v:,.0f}")
            if 'Amp Ratio' in bot5.columns:
                bot5['Amp Ratio'] = bot5['Amp Ratio'].apply(lambda v: f"{v:,.1f}×")
            st.dataframe(bot5, use_container_width=True, hide_index=True, height=220)

        # ── Full stats table ───────────────────────────────────────────────────
        with st.expander("Full amplification stats table"):
            lg_full = (country_df.groupby('league_name')['amp_ratio']
                       .agg(mean='mean', median='median', count='count', std='std')
                       .round(2).sort_values('mean', ascending=False))
            lg_full.columns = ['Mean ×','Median ×','Count','Std']
            lg_full['Mean ×']   = lg_full['Mean ×'].apply(lambda v: f"{v:.2f}×")
            lg_full['Median ×'] = lg_full['Median ×'].apply(lambda v: f"{v:.2f}×")
            lg_full['Std']      = lg_full['Std'].apply(lambda v: f"{v:.2f}")
            st.dataframe(lg_full, use_container_width=True, height=350)

        st.markdown(f"""<div class="insight-box">
          <b>{selected_country}</b> has <b>{len(country_df):,}</b> players in the dataset with a median
          amplification of <b>{country_df['amp_ratio'].median():.1f}×</b> and a home minimum wage of
          <b>${mw_monthly:,.2f}/month</b>.
          The spread between top and bottom players shows how league placement drives amplification
          far more than skill rating alone for this nationality.
        </div>""", unsafe_allow_html=True)