import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tempfile

st.set_page_config(layout="wide")
st.title("📊 Expectancy Change Viewer")

@st.cache_data

def load_data():
    st.write("📥 Downloading file from Google Drive...")
    url = "https://drive.google.com/uc?export=download&id=1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk"
    try:
        response = requests.get(url)
        response.raise_for_status()
        st.success("✅ File downloaded successfully.")
        df = pd.read_parquet(BytesIO(response.content))
        st.success(f"✅ Parquet loaded: {len(df):,} rows")
        df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce')
        return df.dropna(subset=['EVENT_START_TIMESTAMP'])
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

# -- Sidebar Filters -- #
st.sidebar.header("🔎 Filters")
exp_options = [
    "Favourite Goals", "Underdog Goals", "Total Goals",
    "Favourite Corners", "Underdog Corners", "Total Corners",
    "Favourite Yellow", "Underdog Yellow", "Total Yellow"
]
selected_exp = st.sidebar.multiselect("Select Expectancy Type(s)", exp_options, max_selections=6)

min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

def label_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return "Strong Favourite"
    elif diff > 0.5:
        return "Medium Favourite"
    else:
        return "Slight Favourite"

df['FAVOURITISM_LEVEL'] = df.apply(label_favouritism, axis=1)
fav_filter = st.sidebar.multiselect("Favouritism Level (Goals)", ["Strong Favourite", "Medium Favourite", "Slight Favourite"], default=["Strong Favourite", "Medium Favourite", "Slight Favourite"])

def label_scoreline(row):
    home_goals = row['GOALS_HOME']
    away_goals = row['GOALS_AWAY']
    fav_is_home = row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY']
    if home_goals == away_goals:
        return "Scores Level"
    elif (home_goals > away_goals and fav_is_home) or (away_goals > home_goals and not fav_is_home):
        return "Favourite Winning"
    else:
        return "Underdog Winning"

df['SCORELINE_LABEL'] = df.apply(label_scoreline, axis=1)
score_filter = st.sidebar.multiselect("Scoreline (Goals)", ["Favourite Winning", "Underdog Winning", "Scores Level"], default=["Favourite Winning", "Underdog Winning", "Scores Level"])

# -- Time Bins -- #
time_bins = [(i, i + 5) for i in range(0, 90, 5)]
time_labels = [f"{start}-{end}" for start, end in time_bins]

def compute_exp_change(df, exp_type):
    exp_map = {
        "Goals": ("GOAL_EXP_HOME", "GOAL_EXP_AWAY"),
        "Corners": ("CORNERS_EXP_HOME", "CORNERS_EXP_AWAY"),
        "Yellow": ("YELLOW_CARDS_EXP_HOME", "YELLOW_CARDS_EXP_AWAY")
    }
    metric = [k for k in exp_map if k in exp_type][0]
    col_home, col_away = exp_map[metric]

    df_sorted = df.sort_values(['SRC_EVENT_ID', 'MINUTES'])
    output = []

    for event_id, group in df_sorted.groupby('SRC_EVENT_ID'):
        first_min = group['MINUTES'].min()
        base_row = group.loc[group['MINUTES'] == first_min].iloc[0]
        base_home = base_row[col_home]
        base_away = base_row[col_away]

        fav_is_home = base_row['GOAL_EXP_HOME'] > base_row['GOAL_EXP_AWAY']

        prev_home, prev_away = base_home, base_away
        for _, row in group.iterrows():
            band = next((label for (start, end), label in zip(time_bins, time_labels) if start <= row['MINUTES'] < end), "85-90")

            curr_home = row[col_home]
            curr_away = row[col_away]
            total_base = base_home + base_away
            total_curr = curr_home + curr_away

            fav_curr = curr_home if fav_is_home else curr_away
            fav_base = base_home if fav_is_home else base_away
            dog_curr = curr_away if fav_is_home else curr_home
            dog_base = base_away if fav_is_home else base_home

            include = False
            if "Favourite" in exp_type and fav_curr != prev_home if fav_is_home else prev_away:
                delta = fav_curr - fav_base
                include = True
            elif "Underdog" in exp_type and dog_curr != prev_away if fav_is_home else prev_home:
                delta = dog_curr - dog_base
                include = True
            elif "Total" in exp_type and (curr_home != prev_home or curr_away != prev_away):
                delta = total_curr - total_base
                include = True

            if include:
                output.append({"Time Band": band, "Change": delta})

            prev_home = curr_home
            prev_away = curr_away

    return pd.DataFrame(output)

# -- Filter -- #
df_filtered = df[
    (df['EVENT_START_TIMESTAMP'].dt.date >= date_range[0]) &
    (df['EVENT_START_TIMESTAMP'].dt.date <= date_range[1]) &
    (df['FAVOURITISM_LEVEL'].isin(fav_filter)) &
    (df['SCORELINE_LABEL'].isin(score_filter))
]

plots = []
if selected_exp:
    n_cols = 2 if len(selected_exp) > 1 else 1
    layout_cols = st.columns(n_cols)

    for i, exp_type in enumerate(selected_exp):
        df_changes = compute_exp_change(df_filtered, exp_type)
        avg_change = df_changes.groupby('Time Band')['Change'].mean().reindex(time_labels, fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
        ax.set_title(f"{exp_type} Expectancy Change\nDate: {date_range[0]} to {date_range[1]} | Fav: {', '.join(fav_filter)} | Scoreline: {', '.join(score_filter)}")
        ax.set_ylabel("Avg Change")
        ax.set_xlabel("Time Band (Minutes)")
        ax.grid(True)

        with layout_cols[i % n_cols]:
            st.pyplot(fig, use_container_width=True)
            plots.append(fig)

    st.markdown("*Favourites are determined using Goal Expectancy at the earliest available minute in each match*")

    if st.button("Download All Charts as PDF"):
        pdf = FPDF()
        for fig in plots:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=190)
        pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="expectancy_charts.pdf")
else:
    st.warning("Please select at least one expectancy type to display charts.")
