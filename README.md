# Russia: Wikipedia Sentiment Analysis Web App

This project analyzes sentiment in text from Russia's Wikipedia page using machine learning and deploys the model as an interactive web application with Streamlit. The system takes user-input sentences and classifies their sentiment as either Positive or Negative.

The primary goal was to scrape textual data related to Russia from Wikipedia, preprocess it, and train a classification model. The project uses TF-IDF vectorization to transform text into numerical features for sentiment prediction. The final, trained model is deployed using Streamlit, providing a simple web interface for users to perform sentiment analysis.


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=nltk&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)

## 🚀 Live Demo
**https://russia-wikipedia-sentiment-analysis-app.streamlit.app/** 

## 🛠️ Technologies Used

The project was built using a combination of tools and libraries for data processing, model training, and deployment:

* **Core Language:** Python 
* **Web Framework:** Streamlit 
* **Machine Learning:** Scikit-learn 
* **Data Manipulation:** Pandas & NumPy 
* **NLP Preprocessing:** NLTK (Natural Language Toolkit) 
* **Model Persistence:** Pickle 

## 📋 Project Workflow

The project was executed in a structured workflow, from data collection to final deployment:

1.  **Data Preprocessing**:
    * **Web Scraping**: Raw text was extracted from Wikipedia articles related to Russia.
    * **Text Cleaning**: Unwanted characters, numbers, symbols, and HTML tags were removed.
    * **Tokenization**: Text was split into individual words for analysis.
    * **Stop-word Removal**: Common but irrelevant words (e.g., "is", "the", "and") were eliminated with word cloud visualization.
    * **TF-IDF Vectorization**: The cleaned text was converted into a numerical feature matrix.

2.  **Sentiment Analysis Model Training**:
    * **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the dataset to ensure an equal representation of Positive and Negative sentiments for training.
    * **Model Training**: The dataset was trained on several machine learning classifiers, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naïve Bayes, and K-Nearest Neighbors (KNN).

3.  **Model Selection and Deployment**:
    * **Best Model**: The Random Forest Classifier was selected as the final model due to its superior performance, achieving an accuracy of 85.1%.
    * **Serialization**: The trained model and TF-IDF vectorizer were saved using Pickle.
    * **Deployment**: A Streamlit web application was built to provide a user-friendly interface for real-time sentiment predictions.

## 📊 Dataset

The dataset was sourced by scraping Wikipedia articles about Russia.

### Initial Data Distribution

Initially, the dataset was imbalanced, with a large number of Neutral sentences.

| Sentiment | Count |
| :-------- | :---- |
| Positive  | 619   |
| Negative  | 190   |
| Neutral   | 1954  |

### Processed Data Distribution

To create a balanced dataset for effective training, the "Neutral" category was removed, and SMOTE was applied to oversample the minority "Negative" class. This resulted in a dataset with an equal number of Positive and Negative samples, preventing model bias.

| Sentiment | Count |
| :-------- | :---- |
| Positive  | 619   |
| Negative  | 619   |

## 🤖 Model Performance

Several machine learning models were trained and evaluated. The Random Forest classifier achieved the highest accuracy and was chosen for deployment.

| Model               | Accuracy |
| :------------------ | :------- |
| **Random Forest** | **85.1%**|
| Gradient Boosting   | 82.7%    |
| Logistic Regression | 81.4%    |
| Decision Tree       | 78.3%    |
| KNN                 | 72.8%    |

## 🖥️ Web Application Features

The Streamlit web app offers an intuitive interface for sentiment analysis:

* **User Input**: A text box allows users to enter a sentence for analysis.
* **Real-Time Prediction**: Classifies the input text as "Positive" or "Negative" instantly.
* **Probability Scores**: Displays the model's confidence level for the prediction.
* **Data Visualization**: A bar chart visually represents the sentiment probabilities.

## 📁 File Structure

Russia-Wikipedia-Sentiment-Analysis-App/

├── project_code.ipynb       # Jupyter Notebook for model training 

├── app.py                   # Streamlit web app code 

├── tfidf_vectorizer.pkl     # Saved TF-IDF Vectorizer 

├── sentiment_model.pkl      # Saved Random Forest ML Model 

└── requirements.txt         # Required libraries for installation

## ⚙️ How to Run the Project Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HarshBang/Russia-Wikipedia-Sentiment-Analysis-App.git
    cd Russia-Wikipedia-Sentiment-Analysis-App
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  Open your web browser and navigate to the local URL provided (e.g., `http://localhost:8501`).

## ⚠️ Challenges and Solutions

1.  **Class Imbalance**:
    * **Challenge**: The dataset was heavily skewed towards the Neutral sentiment.
    * **Solution**: The Neutral data was removed, and SMOTE was used to balance the Positive and Negative classes, leading to a more robust model.

2.  **Feature Name Mismatch**:
    * **Challenge**: The Random Forest model initially threw errors due to missing feature names during prediction.
    * **Solution**: The data was explicitly converted into a Pandas DataFrame before being fed into the model to preserve feature names and structure.


## 📈 Future Improvements

* **Expand Dataset**: Incorporate text from more diverse sources beyond Wikipedia to improve the model's ability to generalize.
* **Add Neutral Classification**: Enhance the model to perform three-way classification (Positive, Negative, Neutral).
* **Improve Model Performance**: Experiment with advanced deep learning models such as LSTMs or Transformers to potentially increase accuracy.
* **Enhance User Experience**: Add new visualizations, such as charts that display sentiment trends over time.
