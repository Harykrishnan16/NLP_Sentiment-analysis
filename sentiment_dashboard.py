"""
AI Echo — Sentiment Insights Dashboard (Streamlit)
=====================================================
Run this AFTER sentiment_analysis_model.py has produced files in ./outputs/

    streamlit run sentiment_dashboard.py

Answers the "Key Questions for Sentiment Analysis" from the assignment brief:
    1. Overall sentiment of user reviews
    2. Sentiment vs rating (mismatches)
    3. Keywords/phrases per sentiment class
    4. Sentiment trend over time
    5. Verified vs non-verified sentiment
    6. Review length vs sentiment
    7. Sentiment by location
    8. Sentiment by platform (Web vs Mobile / store)
    9. Sentiment by ChatGPT version
    10. Most common negative feedback themes
Plus a live "Predict Sentiment" box for new review text.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_reviews.csv")

st.set_page_config(page_title="AI Echo — Sentiment Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["cleaned_review"] = df["cleaned_review"].fillna("")
    return df


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    classes = joblib.load(os.path.join(MODEL_DIR, "sentiment_classes.pkl"))
    return model, vectorizer, model_name, classes


def check_artifacts_exist():
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append(DATA_PATH)
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        missing.append(os.path.join(MODEL_DIR, "best_model.pkl"))
    if missing:
        st.error(
            "Required files not found: " + ", ".join(missing) +
            "\n\nPlease run `python sentiment_analysis_model.py` first to generate them."
        )
        st.stop()


def simple_clean(text: str) -> str:
    """Lightweight cleaner for live user input (mirrors the training-time
    cleaning closely enough for TF-IDF scoring without needing NLTK here)."""
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


check_artifacts_exist()
df = load_data()
model, vectorizer, model_name, classes = load_model()

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}

st.title("🤖 AI Echo — ChatGPT Review Sentiment Dashboard")
st.caption(f"Model in use: **{model_name}** (TF-IDF + classifier trained on rating-derived sentiment labels)")

tab_labels = [
    "1️⃣ Overall Sentiment", "2️⃣ Sentiment vs Rating", "3️⃣ Keywords by Sentiment",
    "4️⃣ Sentiment Over Time", "5️⃣ Verified Users", "6️⃣ Length vs Sentiment",
    "7️⃣ By Location", "8️⃣ By Platform", "9️⃣ By Version", "🔟 Negative Themes",
    "🔮 Predict"
]
tabs = st.tabs(tab_labels)

# ---------------------------------------------------------------------------
# 1. Overall sentiment
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("What is the overall sentiment of user reviews?")
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    proportions = (counts / counts.sum() * 100).round(1)

    col1, col2, col3 = st.columns(3)
    col1.metric("😊 Positive", f"{int(counts['Positive'])}", f"{proportions['Positive']}%")
    col2.metric("😐 Neutral", f"{int(counts['Neutral'])}", f"{proportions['Neutral']}%")
    col3.metric("😞 Negative", f"{int(counts['Negative'])}", f"{proportions['Negative']}%")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[SENTIMENT_COLORS[s] for s in counts.index],
    )
    ax.set_title("Overall Sentiment Proportions")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 2. Sentiment vs rating
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("How does sentiment vary by rating?")
    st.caption("Since sentiment is derived from rating (1-2=Negative, 3=Neutral, 4-5=Positive), "
               "this confirms the mapping and highlights any edge cases.")
    cross = pd.crosstab(df["rating"], df["sentiment"])
    st.dataframe(cross, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    cross.plot(kind="bar", stacked=True, ax=ax,
               color=[SENTIMENT_COLORS.get(c, "#999") for c in cross.columns])
    ax.set_title("Sentiment Composition by Star Rating")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 3. Keywords by sentiment
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Which keywords are most associated with each sentiment class?")
    sentiment_choice = st.selectbox("Select sentiment class", ["Positive", "Neutral", "Negative"])
    words = " ".join(df.loc[df["sentiment"] == sentiment_choice, "cleaned_review"]).split()

    if words:
        top_words = pd.Series(words).value_counts().head(20)
        st.bar_chart(top_words)

        try:
            from wordcloud import WordCloud
            wc = WordCloud(width=900, height=400, background_color="white").generate(" ".join(words))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except ImportError:
            st.info("Install the `wordcloud` package (`pip install wordcloud`) to see a word cloud here too.")
    else:
        st.info("No reviews found for this sentiment class.")

# ---------------------------------------------------------------------------
# 4. Sentiment over time
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("How has sentiment changed over time?")
    dated = df.dropna(subset=["date"])
    if dated.empty:
        st.info("No valid dates found in the dataset to build a trend chart.")
    else:
        freq = st.radio("Aggregate by", ["Week", "Month"], horizontal=True)
        rule = "W" if freq == "Week" else "M"
        trend = (
            dated.set_index("date")
            .groupby([pd.Grouper(freq=rule), "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
        )
        st.line_chart(trend)

# ---------------------------------------------------------------------------
# 5. Verified vs non-verified
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Do verified users leave more positive or negative reviews?")
    cross = pd.crosstab(df["verified_purchase"], df["sentiment"], normalize="index") * 100
    cross = cross.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0).round(1)
    st.dataframe(cross, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    cross.plot(kind="bar", ax=ax, color=[SENTIMENT_COLORS.get(c, "#999") for c in cross.columns])
    ax.set_ylabel("% of Reviews")
    ax.set_title("Sentiment Share: Verified vs Non-Verified")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 6. Review length vs sentiment
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Are longer reviews more likely to be negative or positive?")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="sentiment", y="word_count", data=df,
        order=["Negative", "Neutral", "Positive"],
        palette=SENTIMENT_COLORS, ax=ax,
    )
    ax.set_ylabel("Word Count")
    ax.set_title("Review Length by Sentiment")
    st.pyplot(fig)

    avg_len = df.groupby("sentiment")["word_count"].mean().reindex(["Negative", "Neutral", "Positive"]).round(1)
    st.write("**Average word count by sentiment:**")
    st.dataframe(avg_len.rename("avg_word_count"))

# ---------------------------------------------------------------------------
# 7. By location
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Which locations show the most positive or negative sentiment?")
    cross = pd.crosstab(df["location"], df["sentiment"], normalize="index") * 100
    cross = cross.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0).round(1)
    cross = cross.sort_values("Positive", ascending=False)
    st.dataframe(cross, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    cross.plot(kind="bar", stacked=True, ax=ax, color=[SENTIMENT_COLORS.get(c, "#999") for c in cross.columns])
    ax.set_ylabel("% of Reviews")
    ax.set_title("Sentiment Share by Location")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 8. By platform
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Is there a difference in sentiment across platforms?")
    cross = pd.crosstab(df["platform"], df["sentiment"], normalize="index") * 100
    cross = cross.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0).round(1)
    st.dataframe(cross, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    cross.plot(kind="bar", ax=ax, color=[SENTIMENT_COLORS.get(c, "#999") for c in cross.columns])
    ax.set_ylabel("% of Reviews")
    ax.set_title("Sentiment Share by Platform")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 9. By version
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Which ChatGPT versions are associated with higher/lower sentiment?")
    cross = pd.crosstab(df["version"], df["sentiment"], normalize="index") * 100
    cross = cross.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0).round(1)
    st.dataframe(cross, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    cross.plot(kind="bar", ax=ax, color=[SENTIMENT_COLORS.get(c, "#999") for c in cross.columns])
    ax.set_ylabel("% of Reviews")
    ax.set_title("Sentiment Share by Version")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 10. Negative feedback themes
# ---------------------------------------------------------------------------
with tabs[9]:
    st.subheader("What are the most common negative feedback themes?")
    neg_words = " ".join(df.loc[df["sentiment"] == "Negative", "cleaned_review"]).split()

    if neg_words:
        top_neg = pd.Series(neg_words).value_counts().head(15)
        st.bar_chart(top_neg)
        st.caption("Recurring terms across negative reviews — useful as a quick pulse on common pain points. "
                   "For deeper theme grouping, consider topic modeling (e.g. LDA) on the negative subset.")

        st.write("**Sample negative reviews:**")
        st.dataframe(
            df.loc[df["sentiment"] == "Negative", ["date", "rating", "review"]].sample(
                min(10, (df["sentiment"] == "Negative").sum()), random_state=1
            ),
            use_container_width=True,
        )
    else:
        st.info("No negative reviews found in the dataset.")

# ---------------------------------------------------------------------------
# Predict sentiment for new review text
# ---------------------------------------------------------------------------
with tabs[10]:
    st.subheader("🔮 Predict Sentiment for a New Review")
    user_review = st.text_area("Paste or type a review:", height=120,
                                placeholder="e.g. The app crashes constantly and support never replies.")

    if st.button("Predict Sentiment"):
        if not user_review.strip():
            st.warning("Please enter some review text.")
        else:
            cleaned = simple_clean(user_review)
            X_input = vectorizer.transform([cleaned])
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            proba_dict = dict(zip(model.classes_, proba))

            color = SENTIMENT_COLORS.get(pred, "#999")
            st.markdown(
                f"### Predicted Sentiment: <span style='color:{color}'>{pred}</span>",
                unsafe_allow_html=True,
            )

            proba_df = pd.Series(proba_dict).reindex(["Positive", "Neutral", "Negative"]).fillna(0)
            st.bar_chart(proba_df)
