# Logistic Regression for Algerian Forest Fire Prediction

## Project Overview
This project uses **Logistic Regression** to predict the occurrence of forest fires in Algeria based on the Algerian Forest Fires dataset. The project is implemented in a Jupyter notebook (`Logistic Regression using Algerian Forest Fire dataset.ipynb`) using Python and libraries such as scikit-learn, pandas, numpy, seaborn, and matplotlib. The notebook includes data loading, cleaning, exploratory data analysis (EDA), preprocessing, model training, evaluation, and model persistence using pickling.

The goal is to classify whether a fire occurred (`fire`) or not (`not fire`) based on meteorological and fire weather index features. The model achieves an accuracy of 96% on the test set, with detailed performance metrics including precision, recall, F1-score, and confusion matrix.

## Dataset
The **Algerian Forest Fires dataset** contains 246 records from two regions in Algeria: Bejaia and Sidi Bel-Abbes, collected between June and September 2012. The dataset includes 14 features:

- **Date-related features**: day, month, year
- **Meteorological features**: Temperature, RH (Relative Humidity), Ws (Wind Speed), Rain
- **Fire Weather Index (FWI) components**: FFMC (Fine Fuel Moisture Code), DMC (Duff Moisture Code), DC (Drought Code), ISI (Initial Spread Index), BUI (Buildup Index), FWI (Fire Weather Index)
- **Target variable**: Classes (binary: `fire` or `not fire`)
- **Derived feature**: region (0 for Bejaia, 1 for Sidi Bel-Abbes)

**Source**:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)
- [Kaggle](https://www.kaggle.com/datasets/abrambeyer/open-access-algerian-forest-fires-dataset-2012)

**Note**: The dataset file (`Algerian_forest_fires_dataset_UPDATE.csv`) is required to run the notebook. Download it from the above sources and place it in the project directory.

## Requirements
To run the notebook, you need the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- statsmodels

Install the dependencies using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/algerian-forest-fire-prediction.git
   cd algerian-forest-fire-prediction
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with the above libraries if needed.)
3. Download the dataset (`Algerian_forest_fires_dataset_UPDATE.csv`) from the UCI repository or Kaggle and place it in the project directory.
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook "Logistic Regression using Algerian Forest Fire dataset.ipynb"
   ```

## Usage
1. Open the Jupyter notebook in your environment.
2. Ensure the dataset file (`Algerian_forest_fires_dataset_UPDATE.csv`) is in the same directory as the notebook.
3. Run the notebook cells sequentially to:
   - Load and clean the dataset
   - Perform exploratory data analysis (EDA)
   - Preprocess the data (e.g., scaling features, encoding target variable)
   - Train and evaluate the Logistic Regression model
   - Save the trained model using pickle
   - Visualize and analyze model performance
4. The trained model is saved as `modelForPrediction.pkl` for future use.
5. Modify the notebook to experiment with different models or hyperparameters if desired.

## Project Structure
- `Logistic Regression using Algerian Forest Fire dataset.ipynb`: Main notebook with the complete workflow.
- `Algerian_forest_fires_dataset_UPDATE.csv`: Dataset file (not included; download from UCI or Kaggle).
- `modelForPrediction.pkl`: Saved Logistic Regression model (generated after running the notebook).
- `README.md`: Project documentation.
- `requirements.txt` (optional): List of required Python libraries.

## Methodology
1. **Data Loading and Cleaning**:
   - Load the dataset using pandas.
   - Remove irrelevant rows (e.g., region headers).
   - Strip whitespace from column names.
   - Add a `region` column to distinguish Bejaia (0) and Sidi Bel-Abbes (1).
2. **Exploratory Data Analysis (EDA)**:
   - Analyze feature distributions and correlations.
   - Visualize data using seaborn and matplotlib.
3. **Data Preprocessing**:
   - Encode the target variable (`Classes`) as binary (0 for `not fire`, 1 for `fire`).
   - Scale numerical features using `StandardScaler`.
   - Split data into training and testing sets using `train_test_split`.
4. **Model Training**:
   - Train a Logistic Regression model using scikit-learn.
   - Save the trained model using pickle for future predictions.
5. **Evaluation**:
   - Compute accuracy (96%), precision (97.7%), recall (95.6%), and F1-score (96.6%).
   - Generate and analyze the confusion matrix.
   - Break down true positives, false positives, true negatives, and false negatives.

## Results
- **Accuracy**: 96% on the test set.
- **Precision**: 97.7% (high proportion of true positive predictions).
- **Recall**: 95.6% (ability to identify most fire occurrences).
- **F1-Score**: 96.6% (balanced measure of precision and recall).
- **Confusion Matrix**:
  - True Positives: 43
  - False Positives: 1
  - False Negatives: 2
  - True Negatives: 33
- The model is saved as `modelForPrediction.pkl` for reuse.

## Future Improvements
- Experiment with other models (e.g., Random Forest, Gradient Boosting, SVM).
- Address potential class imbalance using techniques like SMOTE.
- Perform feature selection to reduce dimensionality and improve model performance.
- Add cross-validation for more robust evaluation.
- Include hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- UCI Machine Learning Repository for providing the dataset.
- Scikit-learn and pandas documentation for implementation guidance.
- Kaggle community for insights and discussions.

## Contact
For questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/your-username/algerian-forest-fire-prediction/issues) or email at your-email@example.com.
