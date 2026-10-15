# 🎬 Movie Recommendation System

A movie recommendation web application built using Python, NumPy, Pandas, Scikit-learn, and Streamlit.

## About the Project

This project explores machine learning-based recommendation systems using the MovieLens dataset.

It uses matrix factorization to learn user and movie representations and predict user preferences.

## Features

- Personalized movie recommendations
- Interactive Streamlit web interface
- Movie selection based on user preferences
- Matrix factorization implemented with NumPy
- Training and testing accuracy evaluation

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit

## Dataset

MovieLens dataset by GroupLens Research.

## How to Run

1. Clone the repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Place the required MovieLens CSV files in the project folder.
4. Run the application:

   `streamlit run app.py`

## Current Limitations

The collaborative filtering model achieves approximately 50% test accuracy on the current experimental split. Recommendations are experimental and may not always match user preferences.

## Future Improvements

- Improve recommendation accuracy
- Implement hybrid recommendation algorithms
- Add movie posters
- Improve the user interface
- Deploy the application online
