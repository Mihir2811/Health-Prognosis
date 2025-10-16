# Health Prognosis - Heart Disease Prediction System

A machine learning-powered web application that predicts the likelihood of heart disease based on patient medical data using multiple ML algorithms and automatically selects the best-performing model.

## Features

- **Multi-Model Comparison**: Evaluates 8 different ML algorithms to find the best performer
- **Interactive Web Interface**: User-friendly Streamlit app for real-time predictions
- **Automated Model Selection**: Automatically saves the best-performing model based on accuracy and F1-score
- **Data Preprocessing**: Includes feature scaling and stratified train-test splitting
- **Comprehensive Evaluation**: Detailed performance metrics and model comparison

## Dataset

The project uses a Heart Attack dataset with 303 records and 14 features:

- **Age**: Patient age
- **Sex**: Gender (0 = Female, 1 = Male)
- **Constrictive pericarditis**: Chest pain type (0-3)
- **Resting Blood Pressure**: Blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting blood sugar**: > 120 mg/dl (0 = No, 1 = Yes)
- **Resting electrocardiographic**: ECG results (0-2)
- **Maximum heart rate**: Maximum heart rate achieved
- **Exercise induced angina**: (0 = No, 1 = Yes)
- **Oldpeak**: ST depression induced by exercise
- **Slope**: Slope of peak exercise ST segment (0-2)
- **Calcium**: Number of major vessels (0-3)
- **Thalassemia**: Blood disorder (0-3)
- **Target**: Heart disease presence (0 = No, 1 = Yes)

## Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Web Interface**: Streamlit
- **Data Visualization**: Available in Jupyter notebook
- **Model Persistence**: joblib

## Project Structure

```
Health-Prognosis/
├── Data/
│   └── Heart_Attack.xlsx          # Dataset
├── Models/
│   ├── best_model.pkl            # Trained best model
│   └── scaler.pkl                # Feature scaler
├── Colab Files/
│   ├── Disease_Prediction.ipynb  # Jupyter notebook analysis
│   └── run.py                    # Notebook runner
├── main.py                       # Model training and comparison
├── host.py                       # Streamlit web application
├── test.py                       # Testing utilities
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## Machine Learning Models

The system evaluates the following algorithms:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (RBF Kernel)**
4. **K-Nearest Neighbors**
5. **Decision Tree Classifier**
6. **Naive Bayes**
7. **Gradient Boosting Classifier**
8. **XGBoost Classifier**

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Health-Prognosis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python main.py
   ```

4. **Launch the web application**
   ```bash
   streamlit run host.py
   ```

## Usage

### Training Models
Run the model comparison script to train all algorithms and save the best performer:
```bash
python main.py
```

### Web Application
1. Start the Streamlit app: `streamlit run host.py`
2. Open your browser to `http://localhost:8501`
3. Enter patient information in the form
4. Click "Predict" to get the heart disease prediction

### Jupyter Notebook
Explore the data analysis and model development process in `Colab Files/Disease_Prediction.ipynb`

## Model Performance

The system automatically:
- Splits data using stratified sampling (80% train, 20% test)
- Applies StandardScaler for feature normalization
- Evaluates models using accuracy and F1-score
- Saves the best-performing model and scaler

## Configuration

Key parameters can be modified in `main.py`:
- Test size ratio
- Random state for reproducibility
- Model hyperparameters
- Evaluation metrics

## Input Features for Prediction

When using the web interface, provide:
- Age (20-100)
- Sex (0 or 1)
- Chest pain type (0-4)
- Resting blood pressure (80-200 mm Hg)
- Cholesterol level (100-400 mg/dl)
- Fasting blood sugar (0 or 1)
- ECG results (0-2)
- Maximum heart rate (60-220)
- Exercise induced angina (0 or 1)
- ST depression (0.0-6.0)
- Slope (0-2)
- Major vessels (0-3)
- Thalassemia (0-3)

## Output

The system provides:
- **Binary prediction**: Heart disease present (1) or absent (0)
- **Model confidence**: Based on the best-performing algorithm
- **Performance metrics**: Accuracy and F1-score for model evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Mihir Panchal**

## Acknowledgments

- Heart disease dataset contributors
- scikit-learn and Streamlit communities
- Open source ML libraries

---

**Note**: This system is for educational and research purposes only. Always consult healthcare professionals for medical decisions.
