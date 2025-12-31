# Titanic Survival Predictor (Decision Tree Analysis)

### Project Overview
This project is an end-to-end machine learning pipeline designed to predict passenger survival based on the 1912 Titanic manifest data. Unlike linear models, this version utilizes a **Decision Tree Classifier** to identify non-linear patterns and survival rules within the historical record.

### Technical Architecture
* **Backend:** FastAPI (Python 3.13)
* **Frontend:** HTML5 / Tailwind CSS (Minimalist Premium Design)
* **Model:** Decision Tree Classifier (Scikit-Learn)
* **Deployment:** Render Cloud Hosting

### Model Performance
After training and validation on the Titanic dataset, the model achieved the following metrics:
* **Accuracy Score:** 78.77%
* **Methodology:** Recursive partitioning based on socio-economic status, gender, and age.

### Data Engineering
1.  **Imputation:** Handled missing chronological data (Age) using the mean value of the dataset.
2.  **Encoding:** Transformed categorical variables (Sex, Port of Embarkation) into numerical features for algorithmic processing.
3.  **Persistence:** The trained pipeline was serialized using `joblib` for real-time inference on the web interface.

### How to Run
1. Enter passenger details (Class, Gender, Age, etc.) into the "Optimal Analysis" interface.
2. Click **Generate Survival Profile**.
3. The system will return a classification (Survived/Did Not Survive) along with a **Confidence Score** calculated via probability estimates.
