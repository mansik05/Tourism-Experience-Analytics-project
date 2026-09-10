# Tourism Experience Analytics

A machine learning project for analyzing tourism data and building prediction and recommendation systems.

## Project Objectives

- Analyze tourism patterns using Exploratory Data Analysis (EDA)
- Predict tourist **VisitMode**
- Predict attraction **Rating**
- Recommend attractions based on historical user preferences
- Deploy the solution using **Streamlit**

## Dataset

The project uses tourism datasets containing information about:

- Users
- Attractions
- Transactions
- Cities
- Countries
- Regions
- Continents
- Attraction Types
- Visit Modes

After preprocessing, the master dataset contains **52,930 tourism transactions and 22 features**.

## Machine Learning

### Classification
**Objective:** Predict `VisitMode`

Models evaluated:
- Logistic Regression
- Random Forest
- Extra Trees

**Best Model:** Extra Trees  
**Accuracy:** 45.23%  
**Weighted F1 Score:** 44.94%

### Regression
**Objective:** Predict attraction `Rating` (1–5)

Models evaluated:
- Linear Regression
- Random Forest
- Extra Trees

**Best Model:** Linear Regression  
**RMSE:** 0.9514  
**R² Score:** 0.0389

### Recommendation
**Method:** Item-Based Collaborative Filtering using Cosine Similarity

**Evaluation:** Leave-One-Out Precision@5  
**Precision@5:** 0.0357

## Project Structure

```text
Tourism Experience Analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       └── master_tourism_data.csv
│
├── models/
│   ├── visit_mode_model.pkl
│   ├── rating_model.pkl
│   ├── user_item_matrix.pkl
│   ├── attraction_similarity.pkl
│   └── attraction_info.pkl
│
├── notebooks/
│   └── tourism_analysis.ipynb
│
└── src/
    ├── classification.py
    ├── regression.py
    └── recommendation.py

Technologies used:
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Joblib
Installation

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
Run the Project

Run classification:
python src/classification.py

Run regression:
python src/regression.py

Run recommendation system:
python src/recommendation.py

Launch the Streamlit application:
streamlit run app.py
Streamlit Application

The application provides:
Tourism analytics dashboard
VisitMode prediction
Rating prediction
Personalized attraction recommendations

Conclusion:-
The project demonstrates the use of data analytics, supervised machine learning, and collaborative filtering to analyze tourism experiences and provide predictive and personalized insights.