import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import datetime
import requests
import os
from transformers import pipeline

# Set up
st.set_page_config(page_title="ESG & Sentiment Dashboard", layout="wide")
st.title("📊 ESG & Sentiment Dashboard for Public Companies")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL").upper()

# Get ESG + stock data
with st.spinner("Fetching data..."):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        esg_data = info.get("esgScores", {})
    except Exception:
        st.error("Unable to fetch ESG data. Try another ticker.")
        st.stop()

    hist = stock.history(period="1y")
    hist["Return"] = hist["Close"].pct_change().fillna(0)

# ESG Display
st.subheader("🌱 ESG Score Breakdown")
if esg_data:
    esg_df = pd.DataFrame(esg_data.items(), columns=["Factor", "Score"])
    st.dataframe(esg_df)
else:
    st.warning("ESG data not available for this ticker.")

# NewsAPI + FinBERT for sentiment
st.subheader("📰 News Sentiment")

newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    st.warning("No NEWSAPI_KEY found in secrets. Add it in Streamlit Cloud settings.")
else:
    today = datetime.datetime.now().date()
    last_week = today - datetime.timedelta(days=7)
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={last_week}&sortBy=popularity&apiKey={newsapi_key}"
    response = requests.get(url).json()

    if response.get("status") == "ok":
        headlines = [article["title"] for article in response["articles"][:5]]
        st.write("Recent headlines:")
        for hl in headlines:
            st.write("•", hl)

        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        sentiment_scores = classifier(headlines)
        df_sent = pd.DataFrame(sentiment_scores)
        df_sent["Headline"] = headlines
        df_sent = df_sent[["Headline", "label", "score"]]
        st.dataframe(df_sent)

        # Sentiment scoring
        sentiment_map = {"positive": 1, "neutral": 0.5, "negative": 0}
        avg_sentiment = df_sent["label"].map(sentiment_map).mean()
    else:
        st.error("Failed to fetch news.")
        avg_sentiment = None

# Final Scoring
st.subheader("📈 ESG + Sentiment Composite Score")

if esg_data and avg_sentiment is not None:
    final_score = (
        esg_data.get("environmentScore", 0) * 0.3 +
        esg_data.get("socialScore", 0) * 0.3 +
        esg_data.get("governanceScore", 0) * 0.2 +
        avg_sentiment * 10 * 0.2
    )
    st.metric("Composite ESG-Sentiment Score", round(final_score, 2))
else:
    st.info("Full score unavailable — missing data.")

# Correlation analysis
st.subheader("📊 Return vs ESG Visualization")

fig = px.line(hist, x=hist.index, y="Return", title="Daily Return (1Y)", labels={"Return": "Return"})
st.plotly_chart(fig)

---

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, yfinance, NewsAPI, FinBERT")
