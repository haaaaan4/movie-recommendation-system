# 🎬 Movie Recommendation & Hit/Flop Prediction System

A machine learning project built using Python and the **MovieLens dataset** that combines a **content-based movie recommendation system** with a **movie Hit/Flop prediction model**.

The application allows users to search for a movie, view its information, predict whether it is likely to be a **Hit or Flop**, and receive recommendations for similar movies.

---

## 📌 Project Overview

The goal of this project is to build a complete movie analysis and recommendation application using movie genres and ratings.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Movie genre encoding
- Movie rating analysis
- Content-based movie recommendations
- Cosine similarity
- Hit/Flop classification
- XGBoost machine learning model
- Interactive Streamlit web application

---

## 🚀 Features

### 🎬 Movie Search

Users can enter a movie name and search the MovieLens dataset.

### ⭐ Movie Information

The application displays:

- Average rating
- Release year
- Movie genres
- Number of genres

### 🤖 Hit/Flop Prediction

The machine learning model predicts whether a movie is likely to be:

- 🎉 **Hit**
- 🎭 **Flop**

The prediction model uses:

- Movie genres
- Average rating

Prediction probabilities are also displayed to show the model's estimated confidence.

### 🍿 Movie Recommendations

The recommendation system finds movies similar to the selected movie using:

- Genre features
- Content-based filtering
- Cosine similarity

### 📊 Data Analysis

The project includes exploratory analysis of:

- Movie ratings
- Rating distributions
- Movie genres
- Movie statistics
- Rating counts

---

## 🧠 Machine Learning

### 1. Movie Recommendation Model

A content-based recommendation approach is used.

Movie genres are transformed into numerical features using **MultiLabelBinarizer**.

Cosine similarity is then used to measure the similarity between movies.

```text
Movie Genres
     ↓
Genre Encoding
     ↓
Feature Vectors
     ↓
Cosine Similarity
     ↓
Similar Movies
📊 Model Evaluation

The Hit/Flop classification model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Because the dataset contains significantly more Flop movies than Hit movies, class balancing was considered during model development to improve the detection of Hit movies.

📌 Project Workflow
MovieLens Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Genre Encoding
       ↓
       ┌───────────────────────┐
       ↓                       ↓
Recommendation Model     Prediction Model
       ↓                       ↓
Cosine Similarity            XGBoost
       ↓                       ↓
Similar Movies             Hit / Flop
       └───────────┬───────────┘
                   ↓
             Streamlit App
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
Joblib
Streamlit
Jupyter Notebook
VS Code
📂 Project Structure
movie-recommendation-prediction/
│
├── app.py
├── movie.ipynb
│
├── movie_hit_flop_model.pkl
├── label_encoder.pkl
├── genre_encoder.pkl
├── movie_data.pkl
├── genre_features.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
File Description
File	Description
app.py	Streamlit web application
movie.ipynb	Main notebook containing data analysis and model development
movie_hit_flop_model.pkl	Trained XGBoost Hit/Flop prediction model
label_encoder.pkl	Target label encoder
genre_encoder.pkl	Movie genre encoder
movie_data.pkl	Processed movie dataset
genre_features.pkl	Genre feature matrix used for recommendations
requirements.txt	Python dependencies
.gitignore	Files excluded from Git
README.md	Project documentation
📊 Dataset

This project uses the MovieLens dataset provided by GroupLens.

Dataset:

https://grouplens.org/datasets/movielens/

The dataset contains movie information, genres, and user ratings.

Note: Large raw dataset files such as ratings.csv may not be included in the GitHub repository because of GitHub file-size limitations. The dataset can be downloaded directly from the MovieLens website.

🔍 Exploratory Data Analysis

The project includes analysis of:

Movie ratings
Rating distributions
Movie genres
Rating counts
Movie statistics
Popularity patterns
Relationships between movie features
▶️ Run the Project Locally
1. Clone the repository
git clone https://github.com/haaaaan4/movie-recommendation-prediction.git
2. Open the project folder
cd movie-recommendation-prediction
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
python -m streamlit run app.py

The application will be available at:

http://localhost:8501
🌐 Application

The Streamlit application provides a simple interface where users can:

Search for a movie
View its rating and genres
Analyze the movie
Get a Hit/Flop prediction
View prediction probabilities
Get similar movie recommendations
🔮 Future Improvements
🤝 Add collaborative filtering
🧠 Build a hybrid recommendation system
🎯 Improve recommendation accuracy
🎞️ Add movie posters
👤 Add personalized recommendations
📚 Add more movie metadata
📈 Further improve prediction performance
🌐 Deploy the application publicly

👨‍💻 Author
Muhammed Alhaan

GitHub:
https://github.com/haaaaan4

⭐ Project

If you find this project interesting, consider giving the repository a ⭐ on GitHub.