import streamlit as st
import pandas as pd
import plotly.express as px
from llmops import LLMSystem  # Your existing system class
import datetime

# Initialize system in session state
if 'system' not in st.session_state:
    st.session_state.system = LLMSystem()

# Sidebar Navigation
st.sidebar.title("🔧 LLMOps Dashboard")
page = st.sidebar.radio("Go to", ["📄 Summarization", "📊 Monitoring", "⚙️ Retraining"])

# ============ 📄 TEXT SUMMARIZATION PAGE ============
if page == "📄 Summarization":
    st.title("📄 AI Summarization System")
    st.write("Enter a passage, and the AI will generate a summary for you.")

    input_text = st.text_area("✍️ Enter text:", height=150)

    # Generate Summary Button
    if st.button("🚀 Generate Summary"):
        if input_text.strip():
            with st.spinner("⏳ Generating summary..."):
                summary = st.session_state.system.process_input(input_text)
                st.session_state.last_summary = summary
                st.rerun()
        else:
            st.warning("⚠️ Please enter some text.")

    # Display Summary & Rating
    if 'last_summary' in st.session_state:
        st.subheader("📝 Generated Summary")
        st.success(st.session_state.last_summary)

        # User Rating
        rating = st.slider("⭐ Rate this summary (1-5)", 1, 5, 3)
        if st.button("✅ Submit Rating"):
            st.session_state.system.monitor.log_interaction(
                input_text, st.session_state.last_summary, rating
            )
            st.success("🎉 Rating submitted!")
            del st.session_state.last_summary
            st.rerun()


# ============ 📊 MONITORING PAGE ============
elif page == "📊 Monitoring":
    st.title("📊 System Monitoring & Insights")
    st.write("View real-time metrics on LLM performance.")

    metrics = st.session_state.system.monitor.calculate_metrics()

    # If no data is available yet
    if not metrics:
        st.warning("No metrics available yet. Process more summaries first.")
    else:
        # Metrics Overview
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Total Interactions", metrics["sample_count"])
        col2.metric("⭐ Average Rating", f"{metrics['avg_rating']:.2f}/5")
        col3.metric("📉 Data Drift", f"{metrics['drift_score']:.2f}")

        st.subheader("📈 Ratings Distribution")
        ratings_df = pd.DataFrame(list(metrics["rating_distribution"].items()), columns=["Rating", "Count"])
        fig_ratings = px.bar(ratings_df, x="Rating", y="Count", color="Rating", title="User Ratings Distribution")
        st.plotly_chart(fig_ratings, use_container_width=True)

        st.subheader("📊 ROUGE Score Over Time")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rouge_scores = {"Date": [timestamp], "ROUGE-L": [metrics["rouge_score"]]}
        df_rouge = pd.DataFrame(rouge_scores)
        fig_rouge = px.line(df_rouge, x="Date", y="ROUGE-L", title="ROUGE Score Trend", markers=True)
        st.plotly_chart(fig_rouge, use_container_width=True)

        # AI-powered Suggestions
        st.subheader("🧠 AI-Based Suggestions")
        if metrics["avg_rating"] < 3.0:
            st.warning("⚠️ Low average rating detected. Consider improving the model!")
        if metrics["drift_score"] > 0.5:
            st.warning("⚠️ High data drift detected. You might need retraining soon.")
        else:
            st.success("✅ Model is performing well. Keep collecting feedback!")

# ============ ⚙️ RETRAINING PAGE ============
elif page == "⚙️ Retraining":
    st.title("⚙️ Model Retraining & Updates")
    st.write("Check if retraining is needed and initiate it manually.")

    rated_samples = len(st.session_state.system.monitor.db.get_rated_samples())

    if rated_samples < 10:
        st.warning(f"⚠️ Not enough rated samples for retraining (Need 25, Have {rated_samples})")
    else:
        st.success(f"✅ Sufficient samples for retraining! ({rated_samples} available)")
        if st.button("🔄 Start Retraining"):
            with st.spinner("Training in progress..."):
                success = st.session_state.system.retrainer.retrain()
                if success:
                    st.success("🎉 Model retrained successfully! New model is active.")
                else:
                    st.error("❌ Retraining failed. Check logs for issues.")
