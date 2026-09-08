# 🤖 AI Echo — ChatGPT Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP) and Machine Learning project** that analyzes ChatGPT user reviews, identifies sentiment patterns, and provides an interactive **Streamlit dashboard** for exploring review insights and predicting the sentiment of new review text.

---

## 📌 Project Overview

**AI Echo** is a sentiment analysis project designed to understand how users feel about ChatGPT based on their reviews.

The project combines:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Machine Learning Classification
- Model Evaluation
- Data Visualization
- Interactive Streamlit Dashboard
- Live Sentiment Prediction

The project follows an end-to-end Data Science workflow from **raw review data to an interactive prediction application**.

---

## 🎯 Objectives

The main objectives of this project are to:

- Analyze the overall sentiment of user reviews
- Identify Positive, Neutral, and Negative reviews
- Understand the relationship between sentiment and rating
- Identify important keywords for each sentiment class
- Analyze sentiment trends over time
- Compare verified and non-verified user sentiment
- Analyze review length across sentiment classes
- Study sentiment by location
- Compare sentiment across platforms
- Analyze sentiment across different ChatGPT versions
- Identify common negative-feedback themes
- Build a model capable of predicting sentiment for new reviews
- Develop an interactive Streamlit dashboard

---

## 🧠 Machine Learning Workflow

```text
Raw Review Dataset
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Sentiment Label Creation
        ↓
TF-IDF Vectorization
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Save Model & Vectorizer
        ↓
Streamlit Dashboard
        ↓
New Review Prediction
````

---

## 🏷️ Sentiment Classification

The project uses three sentiment classes based on review ratings:

| Sentiment   | Rating    |
| ----------- | --------- |
| 🔴 Negative | 1–2 Stars |
| 🟡 Neutral  | 3 Stars   |
| 🟢 Positive | 4–5 Stars |

> **Note:** Since sentiment labels are derived from star ratings, the Sentiment vs Rating analysis primarily validates the labeling relationship rather than independently measuring sentiment.

---

## 🔤 Natural Language Processing

The review text is cleaned before being used for machine learning.

### Text Preprocessing

The preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing non-alphabetic characters
* Removing unnecessary whitespace
* Converting review text into numerical features

### TF-IDF Vectorization

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert review text into numerical vectors that machine learning models can process.

TF-IDF gives greater importance to words that are useful for distinguishing between reviews while reducing the importance of very common words.

The trained vectorizer is saved as:

```text
outputs/models/tfidf_vectorizer.pkl
```

---

# 📊 Streamlit Dashboard

The `sentiment_dashboard.py` file provides an interactive dashboard for exploring the review dataset and generating sentiment predictions.

### 1️⃣ Overall Sentiment

Displays:

* Total Positive reviews
* Total Neutral reviews
* Total Negative reviews
* Percentage distribution of each sentiment

This provides a quick overview of overall user sentiment.

### 2️⃣ Sentiment vs Rating

Shows how sentiment is distributed across different star ratings.

This helps understand the relationship between review ratings and the sentiment labels.

### 3️⃣ Keywords by Sentiment

Users can select:

* Positive
* Neutral
* Negative

The dashboard displays frequently occurring words for the selected sentiment class.

A WordCloud visualization can also be generated when the required package is available.

### 4️⃣ Sentiment Over Time

Analyzes how sentiment changes over time.

Users can view:

* Weekly sentiment trends
* Monthly sentiment trends

### 5️⃣ Verified vs Non-Verified Users

Compares sentiment distribution between:

* Verified users
* Non-verified users

### 6️⃣ Review Length vs Sentiment

Analyzes review length across different sentiment classes.

The dashboard provides:

* Review-length distribution
* Average word count by sentiment

### 7️⃣ Sentiment by Location

Shows the distribution of Positive, Neutral, and Negative reviews across different locations.

### 8️⃣ Sentiment by Platform

Compares sentiment across different platforms, such as:

* Web
* Mobile / Store

### 9️⃣ Sentiment by ChatGPT Version

Analyzes sentiment distribution across different ChatGPT versions.

### 🔟 Negative Feedback Themes

Identifies frequently occurring terms within negative reviews.

This provides an overview of recurring user complaints and potential pain points.

### 🔮 Live Sentiment Prediction

Users can enter a new review into the dashboard.

Example:

```text
The app crashes constantly and support never replies.
```

The application processes the text and displays:

* Predicted sentiment
* Probability of each sentiment class

---

# 🛠️ Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical operations           |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| Scikit-learn | Machine Learning and TF-IDF    |
| Joblib       | Model serialization            |
| Streamlit    | Interactive dashboard          |
| WordCloud    | Keyword visualization          |

---

# 📂 Project Structure

```text
ai-echo-sentiment-analysis/
│
├── sentiment_analysis_model.py
├── sentiment_dashboard.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── cleaned_reviews.csv
    │
    └── models/
        ├── best_model.pkl
        ├── tfidf_vectorizer.pkl
        ├── best_model_name.pkl
        └── sentiment_classes.pkl
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-echo-sentiment-analysis.git
```

Navigate to the project directory:

```bash
cd ai-echo-sentiment-analysis
```

---

## 2. Install Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or install the packages manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit wordcloud
```

---

## 3. Train the Model

Run the model-training script:

```bash
python sentiment_analysis_model.py
```

This generates the cleaned dataset and trained model artifacts inside the `outputs/` directory.

---

## 4. Launch the Streamlit Dashboard

After the model-training script completes:

```bash
streamlit run sentiment_dashboard.py
```

The Streamlit dashboard will open in your browser.

---

# 📦 Model Artifacts

The dashboard uses the following generated files:

```text
outputs/
│
├── cleaned_reviews.csv
│
└── models/
    ├── best_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── best_model_name.pkl
    └── sentiment_classes.pkl
```

These files allow the Streamlit dashboard to load the trained model and make predictions on new review text.

---

# 📈 Data Science Skills Demonstrated

This project demonstrates practical skills in:

* Data Cleaning
* Exploratory Data Analysis
* Natural Language Processing
* Text Preprocessing
* Feature Engineering
* TF-IDF Vectorization
* Machine Learning
* Classification
* Model Evaluation
* Model Serialization
* Data Visualization
* Sentiment Analysis
* Streamlit Development
* Interactive Dashboard Design

---

# 💼 Business Value

Sentiment analysis can help organizations understand large volumes of customer feedback efficiently.

Potential applications include:

* Identifying customer dissatisfaction
* Monitoring product feedback
* Tracking sentiment trends
* Identifying recurring customer complaints
* Comparing user experience across platforms
* Identifying potential product issues
* Supporting customer-experience decisions

Instead of manually reading thousands of comments, stakeholders can use the dashboard to quickly identify important patterns and potential problem areas.

---

# 🔮 Future Improvements

Potential improvements include:

* Hyperparameter tuning
* Cross-validation
* Topic modeling using LDA
* Transformer-based NLP models such as BERT
* Improved text preprocessing
* Independently annotated sentiment labels
* Interactive filtering by date and rating
* Model performance comparison within the dashboard
* Cloud deployment
* Automated model retraining

---

# ⚠️ Important Consideration

The sentiment labels in this project are derived from star ratings.

Therefore, model performance should be interpreted in the context of this labeling approach.

A future version could use an independently annotated dataset where sentiment is manually labeled to provide a stronger evaluation of sentiment classification performance.
