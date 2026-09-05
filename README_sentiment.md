# AI Echo — ChatGPT Review Sentiment Analysis

## 1. Install dependencies
```
pip install -r requirements_sentiment.txt
```
(`nltk` and `wordcloud` are optional — the scripts fall back gracefully with a
built-in stopword list / skip the word cloud if they aren't installed, but
installing them gives better text cleaning and adds the word cloud charts.)

## 2. Folder setup
Put these files together in one folder:
- `sentiment_analysis_model.py`
- `sentiment_dashboard.py`
- `chatgpt_style_reviews_dataset_xlsx_-_Sheet1.csv` (your dataset)

## 3. Run the training pipeline
```
python3 sentiment_analysis_model.py
```
This will:
- Clean the review text (lowercase, remove punctuation/stopwords, lemmatize)
- Derive a sentiment label from the star rating (1-2 = Negative, 3 = Neutral, 4-5 = Positive)
- Run EDA and save 13 charts (including word clouds) to `outputs/eda/`
- Build TF-IDF features and train Naive Bayes, Logistic Regression, and Random Forest
- Print/save accuracy, precision, recall, F1, and AUC-ROC for each model
- Save the best model + TF-IDF vectorizer to `outputs/models/`
- Save the cleaned + labeled dataset to `outputs/cleaned_reviews.csv`

## 4. Launch the dashboard
```
streamlit run sentiment_dashboard.py
```
Opens a browser dashboard covering all 10 "Key Questions" from the brief —
overall sentiment, sentiment vs rating, keywords by sentiment, sentiment over
time, verified vs non-verified, review length vs sentiment, sentiment by
location/platform/version, negative feedback themes — plus a live "Predict
Sentiment" box for typing in a brand-new review.

## Note on labels
The dataset has no ground-truth sentiment column, so sentiment is derived
from the star rating as instructed in the assignment brief. The ML models
are trained to predict this label directly from the review **text** (via
TF-IDF), so they can classify new reviews that don't have a rating attached
— which is what the "Predict Sentiment" dashboard feature does.
