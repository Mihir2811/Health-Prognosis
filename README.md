# Social Piggybacking: Twitter Sentiment Analysis of COVID-19 Tweets

This project analyzes the sentiment of tweets related to COVID-19 to gain insights into public opinion during the pandemic. It leverages Natural Language Processing (NLP) techniques to classify tweets as positive, negative, or neutral.

## Dataset

The project uses the `covid19_tweets.csv` dataset, which is contained within the `covid19_tweets.rar` archive. This dataset includes a collection of tweets and their corresponding metadata.

## Features

* **Data Preprocessing:** Cleans and preprocesses tweet text for sentiment analysis.
* **Sentiment Analysis:** Utilizes machine learning models to classify the sentiment of each tweet.
* **Data Visualization:** Creates visualizations to represent the distribution of sentiments and other insights.
* **Jupyter Notebook:** The entire analysis is documented and implemented in a Jupyter Notebook (`Social_PiggyBacking.ipynb`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mihir2811/Social-PiggyBacking.git](https://github.com/Mihir2811/Social-PiggyBacking.git)
    cd Social-PiggyBacking
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Extract the dataset:**
    Un-archive the `covid19_tweets.rar` file to get the `covid19_tweets.csv` file.

2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

3.  Open and run the `Social_PiggyBacking.ipynb` notebook to see the analysis.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* WordCloud
* Jupyter Notebook
