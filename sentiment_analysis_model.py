"""
AI Echo — ChatGPT Review Sentiment Analysis
=============================================
End-to-end pipeline: data preprocessing -> NLP text cleaning -> EDA ->
TF-IDF feature engineering -> model training (Naive Bayes, Logistic
Regression, Random Forest) -> evaluation -> saving artifacts for the
Streamlit dashboard.

How to run (VS Code terminal):
    pip install pandas numpy matplotlib seaborn scikit-learn joblib nltk wordcloud
    python sentiment_analysis_model.py

Expected input file (same folder as this script, or update DATA_PATH below):
    chatgpt_style_reviews_dataset_xlsx_-_Sheet1.csv

Outputs created:
    outputs/eda/*.png                     -> EDA charts + word clouds
    outputs/models/best_model.pkl         -> trained sentiment classifier (joblib)
    outputs/models/tfidf_vectorizer.pkl   -> fitted TF-IDF vectorizer
    outputs/model_comparison.csv          -> metrics for all trained models
    outputs/cleaned_reviews.csv           -> cleaned + labeled dataset (project deliverable)

Note on labels: the dataset has no ground-truth sentiment column, so
sentiment labels are derived from the star rating (1-2 = Negative,
3 = Neutral, 4-5 = Positive) per the assignment brief. The ML models are
then trained to predict this label directly from the review TEXT, so the
dashboard can classify brand-new reviews that have no rating attached.
"""

import os
import re
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # safe backend for headless / VS Code runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "chatgpt_style_reviews_dataset_xlsx_-_Sheet1.csv"
OUTPUT_DIR = "outputs"
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

sns.set_theme(style="whitegrid")


def ensure_dirs():
    for d in (OUTPUT_DIR, EDA_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# NLP setup (stopwords + lemmatizer), with offline-safe fallbacks
# ---------------------------------------------------------------------------
FALLBACK_STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for
with at by from up down out off over under again further then once here
there when where why how all any both each few more most other some such
no nor not only own same so than too very s t can will just don should now
this that these those i me my myself we our ours you your yours he him his
she her hers it its they them their as it's
""".split())


def get_nlp_tools():
    """Try to use NLTK stopwords + WordNetLemmatizer; fall back to a small
    built-in stopword list and no-op lemmatizer if NLTK/its data isn't
    available, so the script never crashes."""
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(
                    f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}"
                )
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

        try:
            stop_words = set(nltk_stopwords.words("english"))
        except LookupError:
            stop_words = FALLBACK_STOPWORDS

        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("test")  # trigger wordnet load / fail fast
            lemmatize_fn = lemmatizer.lemmatize
        except LookupError:
            lemmatize_fn = lambda w: w

        return stop_words, lemmatize_fn

    except ImportError:
        print("NLTK not installed — using a built-in stopword list instead "
              "(pip install nltk for full stopword/lemmatization support).")
        return FALLBACK_STOPWORDS, lambda w: w


STOP_WORDS, LEMMATIZE = get_nlp_tools()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/special characters, remove stopwords,
    and lemmatize. This is the core NLP preprocessing step."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                    # punctuation/digits/special chars
    tokens = text.split()                                     # tokenization
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZE(t) for t in tokens]                   # lemmatization
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing review text or rating (core fields)
    before = len(df)
    df.dropna(subset=["review", "rating"], inplace=True)
    df = df[df["review"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with missing/empty review or rating")

    # Fill missing values for other columns
    for col in ["title", "platform", "language", "location", "version", "verified_purchase"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    # Parse date column; some source rows contain literal "########"
    # (an Excel column-width artifact) which become NaT and are kept as
    # missing rather than dropped, since date is only needed for trend charts.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop exact duplicate reviews
    before = len(df)
    df.drop_duplicates(subset=["review", "username", "rating"], inplace=True)
    print(f"Removed {before - len(df)} duplicate rows")

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering: sentiment label + cleaned text
# ---------------------------------------------------------------------------
def rating_to_sentiment(rating: int) -> str:
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ground-truth sentiment label derived from star rating
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    # Cleaned/normalized review text (core NLP preprocessing)
    print("Cleaning review text (lowercase, remove punctuation/stopwords, lemmatize)...")
    df["cleaned_review"] = df["review"].apply(clean_text)

    # Recompute review length from cleaned text word count (more meaningful
    # than the raw character count already present in the source file)
    df["word_count"] = df["cleaned_review"].apply(lambda t: len(t.split()))

    # Helpful-votes flag used in EDA question 2
    df["is_helpful"] = (df["helpful_votes"] > 10).astype(int)

    print("Engineered features: sentiment, cleaned_review, word_count, is_helpful")
    return df


# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------
def _save_wordcloud(text_blob: str, title: str, filename: str):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print(f"  [skipped] '{title}' word cloud — install the 'wordcloud' package to enable this chart.")
        return

    if not text_blob.strip():
        return

    wc = WordCloud(width=900, height=450, background_color="white", colormap="viridis").generate(text_blob)
    plt.figure(figsize=(9, 4.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(os.path.join(EDA_DIR, filename), bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame):
    print("\nRunning EDA, saving charts to:", EDA_DIR)

    # 1. Rating distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="rating", data=df, order=[1, 2, 3, 4, 5], palette="Set2")
    plt.title("Distribution of Review Ratings")
    plt.savefig(os.path.join(EDA_DIR, "rating_distribution.png"), bbox_inches="tight")
    plt.close()

    # Sentiment distribution (from rating-derived label)
    plt.figure(figsize=(5, 4))
    sns.countplot(x="sentiment", data=df, order=SENTIMENT_ORDER, palette="Set2")
    plt.title("Sentiment Distribution")
    plt.savefig(os.path.join(EDA_DIR, "sentiment_distribution.png"), bbox_inches="tight")
    plt.close()

    # 2. Helpful votes (threshold > 10)
    plt.figure(figsize=(4, 4))
    df["is_helpful"].map({1: "Helpful (>10 votes)", 0: "Not marked helpful"}).value_counts().plot(
        kind="pie", autopct="%1.1f%%", ylabel=""
    )
    plt.title("Reviews Marked Helpful (>10 votes)")
    plt.savefig(os.path.join(EDA_DIR, "helpful_votes.png"), bbox_inches="tight")
    plt.close()

    # 3. Word clouds: positive (4-5 star) vs negative (1-2 star)
    positive_text = " ".join(df.loc[df["rating"] >= 4, "cleaned_review"])
    negative_text = " ".join(df.loc[df["rating"] <= 2, "cleaned_review"])
    _save_wordcloud(positive_text, "Common Words in Positive Reviews (4-5 stars)", "wordcloud_positive.png")
    _save_wordcloud(negative_text, "Common Words in Negative Reviews (1-2 stars)", "wordcloud_negative.png")

    # 4. Average rating over time
    if df["date"].notna().any():
        trend = df.dropna(subset=["date"]).set_index("date").resample("W")["rating"].mean()
        plt.figure(figsize=(8, 4))
        trend.plot(marker="o")
        plt.title("Average Rating Over Time (Weekly)")
        plt.ylabel("Average Rating")
        plt.savefig(os.path.join(EDA_DIR, "rating_trend.png"), bbox_inches="tight")
        plt.close()
    else:
        print("  [skipped] rating trend chart — no valid dates in the data.")

    # 5. Ratings by location
    plt.figure(figsize=(8, 4))
    loc_avg = df.groupby("location")["rating"].mean().sort_values(ascending=False)
    sns.barplot(x=loc_avg.index, y=loc_avg.values, palette="Set2")
    plt.title("Average Rating by Location")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(EDA_DIR, "rating_by_location.png"), bbox_inches="tight")
    plt.close()

    # 6. Platform comparison
    plt.figure(figsize=(7, 4))
    plat_avg = df.groupby("platform")["rating"].mean().sort_values(ascending=False)
    sns.barplot(x=plat_avg.index, y=plat_avg.values, palette="Set2")
    plt.title("Average Rating by Platform")
    plt.ylabel("Average Rating")
    plt.savefig(os.path.join(EDA_DIR, "rating_by_platform.png"), bbox_inches="tight")
    plt.close()

    # 7. Verified vs non-verified
    plt.figure(figsize=(5, 4))
    ver_avg = df.groupby("verified_purchase")["rating"].mean()
    sns.barplot(x=ver_avg.index, y=ver_avg.values, palette="Set2")
    plt.title("Average Rating: Verified vs Non-Verified")
    plt.ylabel("Average Rating")
    plt.savefig(os.path.join(EDA_DIR, "rating_by_verified.png"), bbox_inches="tight")
    plt.close()

    # 8. Review length per rating
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="rating", y="word_count", data=df, palette="Set2")
    plt.title("Review Word Count by Rating")
    plt.savefig(os.path.join(EDA_DIR, "length_by_rating.png"), bbox_inches="tight")
    plt.close()

    # 9. Most mentioned words in 1-star reviews (bar chart)
    one_star_words = " ".join(df.loc[df["rating"] == 1, "cleaned_review"]).split()
    if one_star_words:
        top_words = pd.Series(one_star_words).value_counts().head(15)
        plt.figure(figsize=(7, 5))
        sns.barplot(x=top_words.values, y=top_words.index, palette="Reds_r")
        plt.title("Top Words in 1-Star Reviews")
        plt.xlabel("Frequency")
        plt.savefig(os.path.join(EDA_DIR, "top_words_1star.png"), bbox_inches="tight")
        plt.close()

    # 10. Average rating by version
    plt.figure(figsize=(7, 4))
    ver_rating = df.groupby("version")["rating"].mean().sort_values(ascending=False)
    sns.barplot(x=ver_rating.index, y=ver_rating.values, palette="Set2")
    plt.title("Average Rating by ChatGPT Version")
    plt.ylabel("Average Rating")
    plt.savefig(os.path.join(EDA_DIR, "rating_by_version.png"), bbox_inches="tight")
    plt.close()

    print("EDA charts saved.")


# ---------------------------------------------------------------------------
# 4. Train & evaluate sentiment classification models
# ---------------------------------------------------------------------------
def evaluate_model(name, y_test, y_pred, y_proba, classes):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    # Multiclass AUC-ROC (one-vs-rest, macro-averaged)
    try:
        y_test_bin = label_binarize(y_test, classes=classes)
        metrics["AUC-ROC"] = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        metrics["AUC-ROC"] = np.nan

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}" if pd.notna(v) else f"{k}: N/A")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics


def plot_confusion_matrix(name, y_test, y_pred, classes):
    safe_name = name.lower().replace(" ", "_")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.savefig(os.path.join(EDA_DIR, f"confusion_matrix_{safe_name}.png"), bbox_inches="tight")
    plt.close(fig)


def train_models(X_train, X_test, y_train, y_test, classes):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = evaluate_model(name, y_test, y_pred, y_proba, classes)
        plot_confusion_matrix(name, y_test, y_pred, classes)

        results.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    return results_df, fitted_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    # Save cleaned + labeled dataset (project deliverable)
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_reviews.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")

    run_eda(df)

    print("\nSentiment label distribution:")
    print(df["sentiment"].value_counts())

    X_text = df["cleaned_review"]
    y = df["sentiment"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    # TF-IDF feature engineering
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF feature matrix: {X_train.shape[1]} features")

    classes = SENTIMENT_ORDER
    results_df, fitted_models = train_models(X_train, X_test, y_train, y_test, classes)

    print("\n=== Model Comparison (sorted by F1-Score) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Save model + vectorizer for the Streamlit dashboard
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))
    joblib.dump(classes, os.path.join(MODEL_DIR, "sentiment_classes.pkl"))
    print(f"\nSaved trained model to {os.path.join(MODEL_DIR, 'best_model.pkl')}")
    print("Done. Run 'streamlit run sentiment_dashboard.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
