# Health Prognosis: A Machine Learning Approach to Disease Prediction

This project uses machine learning to predict the likelihood of a person having a particular disease based on their symptoms. It serves as a tool for preliminary health assessment, offering a prognosis based on user-inputted symptoms.

## Features

* **Symptom-Based Prediction:** Takes a list of symptoms as input and predicts the most probable disease.
* **Machine Learning Models:** Implements various classification models, including Decision Tree, Random Forest, and Naive Bayes, to provide accurate predictions.
* **Comprehensive Dataset:** Trained on a dataset that maps a wide range of symptoms to different diseases.
* **Jupyter Notebook:** The entire analysis and model development process is documented in a Jupyter Notebook (`Health Prognosis.ipynb`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mihir2811/Health-Prognosis.git](https://github.com/Mihir2811/Health-Prognosis.git)
    cd Health-Prognosis
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

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  Open and run the `Health Prognosis.ipynb` notebook to see the model in action. You can input a list of symptoms and get a disease prognosis.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
