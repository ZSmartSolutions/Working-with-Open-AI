
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LLM Experiment Dashboard",
    page_icon="📊",
    layout="wide"
)

LOG_FILE = Path(__file__).parent / "api_logs.json"


# --------------------------------------------------
# Load experiment logs
# --------------------------------------------------

@st.cache_data
def load_logs(file_modified):
    if not LOG_FILE.exists():
        return []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except json.JSONDecodeError:
        st.error("api_logs.json contains invalid JSON.")
        return []


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

def prepare_data(logs):
    if not logs:
        return pd.DataFrame()

    df = pd.DataFrame(logs)

    # Add missing columns safely
    required_columns = [
        "timestamp",
        "test_name",
        "model",
        "parameters",
        "system_prompt",
        "user_input",
        "assistant_response",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "error"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    df["input_tokens"] = pd.to_numeric(
        df["input_tokens"], errors="coerce"
    ).fillna(0)

    df["output_tokens"] = pd.to_numeric(
        df["output_tokens"], errors="coerce"
    ).fillna(0)

    df["total_tokens"] = (
        df["input_tokens"] + df["output_tokens"]
    )

    df["response_length"] = df[
        "assistant_response"
    ].fillna("").astype(str).str.len()

    df["temperature"] = df["parameters"].apply(
        lambda p: p.get("temperature")
        if isinstance(p, dict) else None
    )

    df["reasoning_effort"] = df["parameters"].apply(
        lambda p: (
            p.get("reasoning", {}).get("effort")
            if isinstance(p, dict)
            and isinstance(p.get("reasoning"), dict)
            else None
        )
    )

    df["verbosity"] = df["parameters"].apply(
        lambda p: p.get("verbosity")
        if isinstance(p, dict) else None
    )

    def get_status(row):
        if row.get("error"):
            return "Failed"

        if row.get("status") == "failed":
            return "Failed"

        if isinstance(row.get("assistant_response"), str) and row["assistant_response"].strip():
            return "Success"

        return "Failed"

    df["status"] = df.apply(get_status, axis=1)

    # Unique identifier for each experiment
    df["experiment_id"] = range(1, len(df) + 1)

    df["experiment_label"] = df.apply(
        lambda row: (
            f'{row["experiment_id"]}. {row["test_name"]}'
            if pd.notna(row["test_name"])
            else f'Experiment {row["experiment_id"]}'
        ),
        axis=1
    )

    return df


# --------------------------------------------------
# Load latest data
# --------------------------------------------------

if not LOG_FILE.exists():
    st.error("api_logs.json was not found.")
    st.stop()

logs = load_logs(LOG_FILE.stat().st_mtime)
df = prepare_data(logs)

st.title("📊 LLM Experiment Dashboard")
st.caption("Monitor, analyze, and compare your LLM experiments.")

if df.empty:
    st.warning("No experiment records found.")
    st.stop()


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filters")

models = sorted(df["model"].dropna().unique().tolist())

selected_models = st.sidebar.multiselect(
    "Select models",
    models,
    default=models
)

test_names = sorted(
    df["test_name"].dropna().unique().tolist()
)

selected_tests = st.sidebar.multiselect(
    "Select experiments",
    test_names,
    default=test_names
)

temperatures = sorted(
    df["temperature"].dropna().unique().tolist()
)

selected_temperatures = st.sidebar.multiselect(
    "Temperature",
    temperatures,
    default=temperatures
)

filtered_df = df[
    df["model"].isin(selected_models)
    & df["test_name"].isin(selected_tests)
]

if temperatures:
    filtered_df = filtered_df[
        filtered_df["temperature"].isin(selected_temperatures)
        | filtered_df["temperature"].isna()
    ]

if filtered_df.empty:
    st.warning("No experiments match your filters.")
    st.stop()


# --------------------------------------------------
# KPI metrics
# --------------------------------------------------

total = len(filtered_df)
successful = (filtered_df["status"] == "Success").sum()
failed = (filtered_df["status"] == "Failed").sum()

input_tokens = int(filtered_df["input_tokens"].sum())
output_tokens = int(filtered_df["output_tokens"].sum())
total_tokens = input_tokens + output_tokens

success_rate = (successful / total * 100) if total else 0

st.subheader("Experiment Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Experiments", total)
col2.metric("Successful", successful)
col3.metric("Failed", failed)
col4.metric("Success Rate", f"{success_rate:.1f}%")
col5.metric("Total Tokens", f"{total_tokens:,}")

st.divider()


# --------------------------------------------------
# Token analytics
# --------------------------------------------------

st.subheader("Token Usage")

col1, col2 = st.columns(2)

with col1:
    st.metric("Input Tokens", f"{input_tokens:,}")

with col2:
    st.metric("Output Tokens", f"{output_tokens:,}")

token_df = filtered_df[
    ["experiment_label", "input_tokens", "output_tokens"]
].melt(
    id_vars="experiment_label",
    var_name="Token Type",
    value_name="Tokens"
)

fig = px.bar(
    token_df,
    x="experiment_label",
    y="Tokens",
    color="Token Type",
    barmode="group",
    title="Input vs Output Tokens per Experiment"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Experiment comparison
# --------------------------------------------------

st.subheader("Parameter Analysis")

col1, col2 = st.columns(2)

with col1:
    if filtered_df["temperature"].notna().any():
        temp_df = filtered_df.dropna(subset=["temperature"])

        fig = px.scatter(
            temp_df,
            x="temperature",
            y="output_tokens",
            color="test_name",
            hover_data=["model"],
            title="Temperature vs Output Tokens"
        )

        st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        filtered_df,
        x="test_name",
        y="response_length",
        color="test_name",
        title="Response Length by Experiment"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Model distribution
# --------------------------------------------------

st.subheader("Model Distribution")

model_counts = (
    filtered_df["model"]
    .value_counts()
    .reset_index()
)

model_counts.columns = ["Model", "Experiments"]

fig = px.pie(
    model_counts,
    names="Model",
    values="Experiments",
    hole=0.4,
    title="Experiments per Model"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Compare individual responses
# --------------------------------------------------

st.divider()
st.subheader("Compare LLM Responses")

experiment_options = filtered_df[
    "experiment_label"
].tolist()

if len(experiment_options) >= 2:

    col1, col2 = st.columns(2)

    with col1:
        first = st.selectbox(
            "First experiment",
            experiment_options,
            index=0
        )

    with col2:
        second = st.selectbox(
            "Second experiment",
            experiment_options,
            index=1
        )

    first_row = filtered_df[
        filtered_df["experiment_label"] == first
    ].iloc[0]

    second_row = filtered_df[
        filtered_df["experiment_label"] == second
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {first_row['test_name']}")
        st.write("Model:", first_row["model"])
        st.write("Parameters:", first_row["parameters"])
        st.write("Output tokens:", int(first_row["output_tokens"]))
        st.markdown("**Assistant Response**")
        st.markdown(str(first_row["assistant_response"]))

    with col2:
        st.markdown(f"### {second_row['test_name']}")
        st.write("Model:", second_row["model"])
        st.write("Parameters:", second_row["parameters"])
        st.write("Output tokens:", int(second_row["output_tokens"]))
        st.markdown("**Assistant Response**")
        st.markdown(str(second_row["assistant_response"]))

else:
    st.info("At least two experiments are needed for comparison.")


# --------------------------------------------------
# Experiment history
# --------------------------------------------------

st.divider()
st.subheader("Experiment History")

history_columns = [
    "experiment_id",
    "timestamp",
    "test_name",
    "model",
    "temperature",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "status"
]

st.dataframe(
    filtered_df[history_columns].sort_values(
        "timestamp", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Detailed experiment records
# --------------------------------------------------

st.subheader("Detailed Experiment Records")

for _, row in filtered_df.iterrows():
    with st.expander(
        f'{row["experiment_id"]}. {row["test_name"]} — {row["status"]}'
    ):
        st.write("System Prompt:")
        st.code(str(row["system_prompt"]))

        st.write("User Input:")
        st.write(row["user_input"])

        st.write("Assistant Response:")
        st.markdown(str(row["assistant_response"]))

        if row.get("error"):
            st.error(str(row["error"]))

        st.json(row["parameters"])

st.caption("Data source: api_logs.json")
