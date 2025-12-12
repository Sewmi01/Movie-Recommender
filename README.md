# Movie Recommendation System

A content-based movie recommendation system built using Python and Machine Learning.  
The app recommends similar movies based on genres using Cosine Similarity and is deployed with Streamlit.

##  Features
- Content-based recommendations
- Cosine similarity
- Streamlit web application

##  Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit

##  Project Structure
MOVIE-RECOMMENDER/
- data/
  - movies.csv
  - ratings.csv
- app.py
- requirements.txt

##  Run the Project
# Create virtual environment
python -m venv venv
# Activate it (Windows)
venv\Scripts\activate
# Install the libraries
pip install pandas numpy scikit-learn streamlit

# Run the Project
streamlit run app.py


