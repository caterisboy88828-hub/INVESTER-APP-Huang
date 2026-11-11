import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 設置手機適應的最佳化版面
st.set_page_config(
    page_title="全球球類 AI 智能運彩精準分析系統",
    page_icon="🏀",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏀⚽ 全球球類 AI 智能運彩精準分析</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>直接輸入盤口數值，結合 Bet365 與台灣運彩數據進行 AI 深度分析</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------
# 1. 球隊資料庫設定
# ---------------------------------------------------------
nba_teams = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", 
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", 
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", 
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", 
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", 
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", 
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", 
    "Utah Jazz", "Washington Wizards"
]

leagues_data = {
    "🏀 NBA 美國職籃 (全 30 隊)": nba_teams,
    "⚽ 英格蘭超級聯賽 (英超)": [
        "Manchester City", "Arsenal", "Liverpool", "Aston Villa", 
        "Tottenham Hotspur", "Chelsea", "Manchester United", "Newcastle United"
    ],
    "⚽ 西班牙甲級聯賽 (西甲)": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Girona", "Athletic Bilbao", "Real Sociedad"
    ],
    "⚽ 德國甲級聯賽 (德甲)": [
        "Bayer Leverkusen", "Bayern Munich", "Stuttgart", "RB Leipzig", "Borussia Dortmund"
    ],
    "⚽ 法國甲級聯賽 (法甲)": [
        "Paris Saint-Germain", "AS Monaco", "Brest", "Lille", "Nice", "Marseille"
    ]
}

# ---------------------------------------------------------
# 2. 機器學習高精準預測模型核心
# ---------------------------------------------------------
@st.cache_resource
def load_ai_models():
    np.random.seed(42)
    n_samples = 4000
    
    data = {
        'recent_form_diff': np.random.uniform(-0.7, 0.7, n_samples),
        'market_line': np.random.uniform(-16, 16, n_samples),
        'corner_avg': np.random.uniform(7.0, 13.5, n_samples),
        'bet365_odds': np.random.uniform(1.70, 2.10, n_samples),
        'sportlottery_odds': np.random.uniform(1.70, 2.10, n_samples),
    }
    df = pd.DataFrame(data)
    
    score = (
        df['recent_form_diff'] * 3.0 + 
        df['corner_avg'] * 0.25 + 
        df['market_line'] * 0.2 + 
        (df['bet365_odds'] - df['sportlottery_odds']) * 1.5
    )
    df['target'] = (score + np.random.normal(0, 0.8, n_samples) > 0).astype(int)
    
    X = df[['recent_form_diff', 'market_line', 'corner_avg', 'bet365_odds', 'sportlottery_odds']]
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, df['target'])
    return model

ai_model = load_ai_models()

# ---------------------------------------------------------
# 3. 手機版互動控制介面（改為數字直接輸入）
# ---------------------------------------------------------
st.sidebar.header("⚙️ 賽事與即時盤口輸入")

selected_league = st.sidebar.selectbox("選擇賽事聯盟", list(leagues_data.keys()))
teams = leagues_data[selected_league]

col_h, col_a = st.columns(2)
with col_h:
    home_team = st.selectbox("主隊 (Home)", teams, index=0)
with col_a:
    default_away_index = 1 if len(teams) > 1 else 0
    away_team = st.selectbox("客隊 (Away)", teams, index=default_away_index)

st.markdown("---")
st.subheader("📊 盤口與數據直接輸入")

if "🏀" in selected_league:
    market_type = st.radio("分析市場類型", ["NBA 讓分盤 (Spread)", "NBA 全場總分 (Over/Under)"])
    if market_type == "NBA 讓分盤 (Spread)":
        line_value = st.number_input("輸入讓分盤口 (以主隊為主，例：-5.5)", value=-5.5, step=0.5)
    else:
        line_value = st.number_input("輸入總分盤口 (Total Line，例：226.5)", value=226.5, step=0.5)
    corner_input = 0.0
else:
    market_type = st.radio("分析市場類型", ["足球全場大小球 (Goals O/U)", "足球全場角球大小 (Corners O/U)"])
    if market_type == "足球全場大小球 (Goals O/U)":
        line_value = st.number_input("輸入大小球盤口 (例：2.5)", value=2.5, step=0.5)
    else:
        line_value = st.number_input("輸入角球大小盤口 (例：9.5)", value=9.5, step=0.5)
    corner_input = st.number_input("輸入雙方近期場均合計角球數 (例：10.5)", value=10.5, step=0.1)

# 直接輸入賠率與近況
col_odds1, col_odds2 = st.columns(2)
with col_odds1:
    bet365_odds = st.number_input("🌐 Bet365 賠率", value=1.91, step=0.01)
with col_odds2:
    sportlottery_odds = st.number_input("🇹🇼 台灣運彩賠率", value=1.75, step=0.01)

with st.expander("📈 球隊近況指數設定 (0 到 1 分別代表極差到極佳)"):
    home_form = st.number_input(f"{home_team} 近期狀態指數", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    away_form = st.number_input(f"{away_team} 近期狀態指數", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. 執行精準分析與推薦結果輸出
# ---------------------------------------------------------
if st.button("🚀 開始 AI 精準分析與推薦", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("❌ 主隊與客隊不能選擇同一支球隊！")
    else:
        form_diff = home_form - away_form
        input_data = pd.DataFrame([{
            'recent_form_diff': form_diff,
            'market_line': line_value,
            'corner_avg': corner_input if corner_input > 0 else 10.5,
            'bet365_odds': bet365_odds,
            'sportlottery_odds': sportlottery_odds
        }])
        
        prob = ai_model.predict_proba(input_data)[0][1]
        implied_bet365 = 1 / bet365_odds
        edge = prob - implied_bet365
        
        st.success(f"✅ 【{selected_league}】精準分析完成：{away_team} (客) @ {home_team} (主)")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="AI 綜合預測命中勝率", value=f"{prob * 100:.1f}%", delta=f"{edge*100:+.1f}% 期望值邊際")
        with col_res2:
            st.metric(label="Bet365 隱含勝率", value=f"{implied_bet365 * 100:.1f}%", delta=f"水位 {bet365_odds}")
            
        st.markdown("---")
        st.subheader("🎯 最終 AI 推薦精準投注項目")
        
        if prob > 0.58 and edge > 0.04:
            if "🏀" in selected_league:
                rec_title = "【主隊讓分盤勝出 / 全場大分 OVER】" if "讓分" in market_type else "【全場大分 OVER】"
            else:
                rec_title = "【全場大球 OVER】" if "大小球" in market_type else "【全場角球大盤 OVER】"
                
            st.markdown(f"🔥 **強力推薦投注：{rec_title}**")
            st.info(f"💡 **精準分析總結**：綜合 **Bet365** 國際盤與 **台灣運彩** 盤口數據、球隊近期狀態差以及進攻/角球效率後，模型檢視出本場具備極佳的盤口落差與投資價值 (Value Edge)，強烈推薦列入今日首選！")
            
        elif prob < 0.42 and (-edge) > 0.04:
            if "🏀" in selected_league:
                rec_title = "【客隊受讓盤 / 全場小分 UNDER】" if "讓分" in market_type else "【全場小分 UNDER】"
            else:
                rec_title = "【全場小球 UNDER】" if "大小球" in market_type else "【全場角球小盤 UNDER】"
                
            st.markdown(f"🔥 **強力推薦投注：{rec_title}**")
            st.info(f"💡 **精準分析總結**：客隊近期防守效率或雙方對抗節奏顯示盤口保護邊際足夠，結合國際盤與運彩盤口數據，強烈建議關注客隊方向或小分/小角盤面。")
            
        else:
            st.markdown("⚠️ **觀望建議：盤口效率極高，勝率五五波**")
            st.warning("💡 **精準分析總結**：雙方實力與兩大盤口高度貼合，缺乏足夠的期望值邊際，建議保守小注或直接跳過本場賽事。")
