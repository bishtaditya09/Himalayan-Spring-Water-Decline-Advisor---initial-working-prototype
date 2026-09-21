import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generation import generate_dataset
from forecasting import forecast_spring, forecast_district_summary
from rag_pipeline import SpringAdvisoryRAG

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "springs_synthetic.csv")

st.set_page_config(page_title="Himalayan Spring Water Decline Advisor", page_icon="💧", layout="wide")


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_dataset()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@st.cache_resource
def load_rag():
    return SpringAdvisoryRAG()


df = load_data()
rag = load_rag()

st.title("💧 Himalayan Spring Water Decline Advisor")
st.caption(
    "SDG 6 (Clean Water & Sanitation) · SDG 13 (Climate Action) · SDG 15 (Life on Land) — "
    "AI for Sustainability Project"
)

with st.expander("ℹ️ About this project & data", expanded=False):
    st.markdown(
        """
**The real problem:** Himalayan springs — the primary drinking water source for most
hill villages in Uttarakhand — are declining at scale. This isn't a hypothetical issue;
it's documented in NITI Aayog's 2018 report on Himalayan springs (widely cited figures
put spring discharge decline in the range of 25–50% across the region), driven by
deforestation, unplanned construction disrupting recharge zones, and shifting rainfall
patterns.

**⚠️ Data disclosure:** Village-level spring discharge records are not publicly available
at the granularity this problem needs. The discharge and rainfall data in this demo are
**synthetically generated**, grounded in the documented trend ranges above (seasonal
monsoon-driven recharge, gradual long-term decline, district-level variation by forest
cover). This is clearly labeled throughout the app — treat forecasts as an illustration
of the *method*, not as real measurements for any specific village.

**Two AI components — and how to tell them apart from raw data in this app:**
1. **Forecasting** — a Holt-Winters time-series model per spring, trained only on
   history up to today, projecting discharge trend and flagging risk level. In
   Tab 2, the **solid blue line is raw synthetic data**; the **dashed red line and
   every "Risk Level" badge you see is the model's output** — none of that is
   pulled from a file, it's computed live by the model at runtime.
2. **Retrieval-augmented advisory** — a grounded Q&A layer over real springshed
   management practices. In Tab 3, typing any question runs it through a TF-IDF
   similarity model against 23 corpus passages and returns the best-matching ones
   with a relevance score — it is not a keyword lookup table.
        """
    )

st.info(
    "🔍 **Quick way to see the AI, not just the app:** In Tab 2, change the spring "
    "dropdown — the risk badge and forecast line recompute instantly because a model "
    "is re-fit each time, not read from a pre-saved answer.",
    icon="🤖",
)

tab1, tab2, tab3 = st.tabs(["📊 District Risk Overview", "📈 Spring Forecast Detail", "💬 Advisory Assistant"])

# ---------------------------------------------------------------------------
# TAB 1: District risk overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("District-Level Risk Summary")
    st.caption(
        "🤖 **This entire table is AI output.** Every 'Risk' badge and 'Forecast Discharge' "
        "value below is computed live by fitting a separate time-series model to each "
        "spring — nothing here is looked up from a static file."
    )
    district = st.selectbox("Select a district", sorted(df["district"].unique()), key="district_overview")

    with st.spinner("🤖 Fitting a forecasting model for every spring in this district..."):
        summary = forecast_district_summary(df, district)

    risk_colors = {"CRITICAL": "🔴", "WATCH": "🟡", "STABLE": "🟢"}
    summary_display = summary.copy()
    summary_display["risk_level"] = summary_display["risk_level"].apply(lambda r: f"{risk_colors[r]} {r}")
    st.dataframe(
        summary_display.rename(columns={
            "spring_id": "Spring", "risk_level": "Risk", "pct_of_baseline": "% of Baseline",
            "forecast_end_avg_lpm": "Forecast Discharge (L/min)"
        }),
        use_container_width=True, hide_index=True,
    )

    n_critical = (summary["risk_level"] == "CRITICAL").sum()
    n_watch = (summary["risk_level"] == "WATCH").sum()
    n_stable = (summary["risk_level"] == "STABLE").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", n_critical)
    c2.metric("🟡 Watch", n_watch)
    c3.metric("🟢 Stable", n_stable)

# ---------------------------------------------------------------------------
# TAB 2: Individual spring forecast detail
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Individual Spring Forecast")
    st.caption(
        "🔵 Solid blue = raw synthetic historical data &nbsp;&nbsp;|&nbsp;&nbsp; "
        "🔴 Dashed red = **AI-generated forecast** (Holt-Winters model output, computed "
        "live — not stored anywhere)"
    )
    col_a, col_b = st.columns(2)
    district2 = col_a.selectbox("District", sorted(df["district"].unique()), key="district_detail")
    springs_in_district = sorted(df[df["district"] == district2]["spring_id"].unique())
    spring_id = col_b.selectbox("Spring", springs_in_district, key="spring_detail")

    with st.spinner("🤖 Fitting time-series model for this spring..."):
        fc = forecast_spring(df, spring_id)

    risk_colors = {"CRITICAL": "🔴", "WATCH": "🟡", "STABLE": "🟢"}
    st.markdown(f"### {risk_colors[fc.risk_level]} AI Risk Assessment: **{fc.risk_level}**")
    st.info(fc.risk_note)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc.history.index, y=fc.history.values, name="Historical discharge (synthetic)",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=fc.forecast.index, y=fc.forecast.values, name="Forecast (next 24 months)",
        line=dict(color="#d62728", dash="dash"),
    ))
    fig.add_hline(y=fc.baseline_avg, line_dash="dot", line_color="gray",
                  annotation_text="2005-2010 baseline avg")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Discharge (litres/min)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Baseline avg (2005-2010)", f"{fc.baseline_avg:.1f} L/min")
    c2.metric("Forecast end avg (next 2 yrs)", f"{fc.forecast_end_avg:.1f} L/min",
               delta=f"{fc.pct_of_baseline - 100:.0f}% vs baseline")

    with st.expander("🤖 See what the model actually did (under the hood)"):
        st.markdown(f"""
- **Model type:** Holt-Winters Exponential Smoothing (trend='add', seasonal='add', period=12)
- **Trained on:** {len(fc.history)} monthly data points for `{fc.spring_id}` only — no other
  spring's data influences this forecast
- **Forecasted:** next {len(fc.forecast)} months, none of which exist in the input data
- **Risk logic:** forecast's last-6-month average ({fc.forecast_end_avg:.1f} L/min) compared
  against this spring's own 2005-2010 baseline ({fc.baseline_avg:.1f} L/min) →
  **{fc.pct_of_baseline:.0f}%** of baseline → classified **{fc.risk_level}**
        """)

# ---------------------------------------------------------------------------
# TAB 3: RAG advisory assistant
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Ask the Springshed Advisory Assistant")
    st.caption(
        "Answers are retrieved directly from a curated corpus of real springshed "
        "management practices — every response is source-attributed, not generated "
        "or guessed."
    )

    example_qs = [
        "How do we revive a spring that is drying up?",
        "Where should recharge pits be dug relative to a spring?",
        "How do we get the community involved and keep the program working long-term?",
        "What is a springshed and how is it mapped?",
    ]
    chosen_example = st.selectbox("Try an example question, or type your own below:",
                                   ["(type my own)"] + example_qs)

    query = st.text_input("Your question:", value="" if chosen_example == "(type my own)" else chosen_example)

    if query:
        with st.spinner("🤖 Running TF-IDF similarity search across 23 corpus passages..."):
            result = rag.answer(query, k=3)
        st.markdown("#### Answer")
        st.write(result["answer"])
        st.markdown("#### Sources & AI relevance scores")
        st.caption(
            "These scores are computed live from your exact query text — type a different "
            "question and they change. This is not a lookup table of pre-written answers."
        )
        for s in result["sources"]:
            st.progress(min(s["relevance"] * 3, 1.0), text=f"📄 {s['title']} — similarity: {s['relevance']}")

    st.divider()
    st.markdown("##### Responsible AI note")
    st.caption(
        "This assistant does not generate free-form advice — it retrieves and surfaces "
        "the most relevant real guidance from its corpus. This avoids the risk of "
        "confidently hallucinated land/water intervention advice, at the cost of being "
        "less conversational than a full LLM. Given the real-world stakes of water "
        "infrastructure decisions, this tradeoff is deliberate."
    )

st.divider()
st.caption(
    "Built for the 1M1B AI for Sustainability Virtual Internship (IBM SkillsBuild & AICTE). "
    "Synthetic data — see 'About this project & data' above for full disclosure."
)
