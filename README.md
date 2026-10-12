# Credit Scoring Model Using Machine Learning

## 📌 Project Overview
This project uses machine learning to predict whether an individual is likely to default on a loan based on their financial history and personal information.

The model learns from financial data and predicts the `defaulted` target column using classification algorithms.

## 🎯 Project Objectives
- Analyze financial and credit-related data.
- Check for missing values in the dataset.
- Prepare features and the target variable.
- Train machine learning classification models.
- Evaluate model performance using Precision, Recall, F1-Score, and ROC-AUC.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## 📂 Dataset Information
The project uses a synthetic credit-scoring dataset containing **1,200 rows and 12 columns**.

The dataset includes the following features:

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `monthly_income` | Monthly income |
| `existing_debt` | Current debt |
| `loan_amount` | Requested or recorded loan amount |
| `credit_history_years` | Length of credit history |
| `missed_payments_12m` | Missed payments in the last 12 months |
| `late_payments_30d` | Late payments over 30 days |
| `late_payments_90d` | Late payments over 90 days |
| `employment_years` | Years of employment |
| `dependents` | Number of dependents |
| `credit_utilization` | Credit utilization level |
| `defaulted` | Target variable indicating loan default |

## ⚙️ Machine Learning Models

### 1. Logistic Regression
Logistic Regression is used as a classification algorithm to predict whether an individual has defaulted.

### 2. Decision Tree Classifier
The Decision Tree Classifier learns decision rules from financial features to classify loan default outcomes.

## 🔄 Project Workflow
1. Import the required Python libraries.
2. Load the dataset using Pandas.
3. Explore the dataset using `head()` and `shape`.
4. Check missing values using `isnull().sum()`.
5. Separate the independent variables (`X`) and dependent variable (`y`).
6. Split the data into training and testing sets using `train_test_split`.
7. Train the Logistic Regression model.
8. Train the Decision Tree Classifier.
9. Evaluate both models using classification metrics.

## 📊 Model Evaluation Results

The following results were obtained in the notebook:

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Precision | 0.5854 | 0.3830 |
| Recall | 0.2581 | 0.3871 |
| F1-Score | 0.3582 | 0.3850 |
| ROC-AUC | 0.5712 | 0.4963 |

These results show that both models have room for improvement. Logistic Regression achieved higher precision and ROC-AUC, while the Decision Tree achieved slightly higher recall and F1-Score.

## 🚀 How to Run the Project
1. Clone or download this repository.
2. Open the notebook in Google Colab.
3. Upload the credit-scoring dataset or update the dataset path.
4. Install the required libraries if needed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
5. Run the notebook cells in order.
6. Review the model evaluation results.

## 📚 Learning Outcomes
- Understanding supervised machine learning.
- Learning classification with Logistic Regression and Decision Trees.
- Separating features and target variables.
- Splitting data into training and testing sets.
- Evaluating classification models using multiple metrics.

## 🔗 Project Notebook
[Open the Credit Scoring Model in Google Colab](https://colab.research.google.com/drive/13jznguL-2OnEm0OHrAR1MlR1oDV-NENG?usp=sharing)

## 👨‍💻 Author
Usama

## ⚠️ Disclaimer
This is an educational machine learning project using synthetic data. Its predictions and evaluation results should not be used to make real-world lending decisions.

##Speech Emotion Recognition Using Deep Learning

This project uses deep learning to classify emotions from speech audio. It processes audio recordings from the RAVDESS dataset, extracts useful audio features, and trains a Convolutional Neural Network (CNN) using TensorFlow and Keras to predict an emotion.

This is a learning project built to practise audio preprocessing, feature extraction, neural networks, and model evaluation.

## Emotions Covered

The model is designed to classify eight emotion categories:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech dataset.

The notebook downloads and extracts the speech audio archive. It then finds `.wav` files and reads the emotion code from each filename.

Dataset source: https://zenodo.org/record/1188976

## Technologies Used

- Python
- NumPy
- Librosa
- TensorFlow
- Keras
- Scikit-learn
- Google Colab

## How It Works

1. **Load the dataset:** Download and extract the RAVDESS audio files.
2. **Extract audio features:** Use Librosa to load audio at 22,050 Hz and extract 40 Mel-frequency cepstral coefficients (MFCCs).
3. **Prepare the data:** Pad or trim each feature sequence to 130 time steps, then convert the features and emotion labels into NumPy arrays.
4. **Split the dataset:** Use an 80/20 training and testing split.
5. **Build the model:** Train a CNN with Conv1D, MaxPooling1D, Flatten, Dense, and Dropout layers.
6. **Train the model:** Train for up to 30 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
7. **Evaluate predictions:** Check test accuracy, a classification report, and a confusion matrix.

## Model Details

- Input features: 130 time steps × 40 MFCC features
- Architecture: 1D Convolutional Neural Network (CNN)
- Output: 8 emotion classes
- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Batch size: 32
- Training epochs: 30

## Results

The notebook's recorded evaluation results are:

- **Test accuracy:** 21.18%
- **Test samples:** 288
- **Training samples:** 1,152

The current model has low classification performance and does not reliably distinguish all eight emotions yet. These results are a baseline, not a claim of a highly accurate emotion detector. Further preprocessing, model tuning, and training experiments are needed.

## How to Run

The notebook was created in Google Colab.

1. Open the project notebook in Google Colab.
2. Run the cells in order.
3. Allow the dataset download and extraction cell to finish.
4. Run the feature extraction, data splitting, model training, and evaluation cells.

### Install dependencies (if running in a separate environment)

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib
```

The notebook downloads the dataset automatically, so a working internet connection is required for that step.

## Project Structure

```text
Speech-Emotion-Recognition/
├── README.md
└── Emotion Recognition from Speech.ipynb
```

Your actual notebook filename or repository structure may differ. Update this section if you use a different filename or add more files.

 Future Improvements

- Improve the model's ability to recognise all eight emotions.
- Experiment with CNN, LSTM, or CNN-LSTM architectures.
- Tune model parameters and compare validation results.
- Review the confusion matrix and per-class precision and recall.
- Test predictions on audio recordings not used during training.

What I Learned

- Loading and preprocessing speech audio
- Extracting MFCC features with Librosa
- Preparing data for a deep learning model
- Building and training a CNN with TensorFlow/Keras
- Evaluating a classifier using accuracy, precision, recall, F1-score, and a confusion matrix

 Author

Created as a hands-on machine learning and deep learning project.

