import streamlit as st
from libraries import *
import openai


df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")


# Step 1: Obtain dataset recommendations from GPT-3
def get_dataset_recommendations(prompt):
    # If no API key is provided, return default dataset recommendations
    default_recommendations = ["Disease_symptom_and_patient_profile_dataset.csv"]
    
    # Call the GPT-3 API to generate dataset recommendations based on the prompt
    # Return the default recommendations if no API key is provided
    if openai.api_key:
        response = openai.Completion.create(
            engine="davinci",  # Choose the appropriate GPT-3 engine
            prompt=prompt,
            max_tokens=50  # Adjust this value as needed
        )
        
        # Extract and return the recommended datasets from the API response
        recommended_datasets = response.choices[0].text.split("\n")
        return recommended_datasets
    else:
        return default_recommendations

# Step 2: Load the recommended dataset
def load_dataset(dataset_path):
    # Load the recommended dataset into a pandas DataFrame
    dataset = pd.read_csv(dataset_path)
    return dataset


def preprocess_data(data):
    LE = LabelEncoder()
    data['Fever'] = LE.fit_transform(data['Fever'])
    data['Cough'] = LE.fit_transform(data['Cough'])
    data['Fatigue'] = LE.fit_transform(data['Fatigue'])
    data['Difficulty Breathing'] = LE.fit_transform(data['Difficulty Breathing'])
    data['Gender'] = LE.fit_transform(data['Gender'])

    data = pd.get_dummies(data, columns=['Blood Pressure', 'Cholesterol Level'], prefix=['BP', 'CL'])

    data['Outcome Variable'] = LE.fit_transform(data['Outcome Variable'])
    # Calculate the frequency of each category in the dataset
    category_counts = data['Disease'].value_counts()

    # Create a new column with the frequency values for each category
    data['Disease_freq'] = data['Disease'].map(category_counts)
    data = data.drop(columns='Disease',axis=1)
       
    return data

df = preprocess_data(df)

# Step 4: Explore the data (EDA)
def perform_eda(data):
    # Conduct exploratory data analysis
    # Use descriptive statistics, data visualization, and statistical analysis
    # to gain insights into the data
    # Plot histograms, scatter plots, correlation matrix, etc.

    # For example:
    # Calculate summary statistics
    summary_stats = data.describe()

    # Create histograms for numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure()
        data[col].hist()
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # Calculate correlation matrix and plot heatmap
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Return the EDA results and insights (you can modify this as needed)
    return {
        'summary_stats': summary_stats,
        'correlation_matrix': correlation_matrix
    }


def identify_feature_engineering_opportunities(data, target):
    # Identify potential feature engineering opportunities based on EDA results
    # This could include transforming existing features, creating new derived features,
    # or selecting relevant features for modeling
    # Return the identified opportunities
    
    # No modifications needed for this step
    return None

# Call the identify_feature_engineering_opportunities function
yVar = df['Outcome Variable']
feature_engineering_opportunities = identify_feature_engineering_opportunities(df, yVar)

# Perform feature engineering
def perform_feature_engineering(data, opportunities):
    # Implement the identified feature engineering techniques on the data
    # This could include feature scaling, one-hot encoding, binning,
    # feature extraction, or other transformations
    # Return the engineered features
    
    # Random Forest Classifier
    RNF = RandomForestClassifier(
        n_estimators=400,
        criterion='gini',
        max_depth=9,
        max_features='sqrt'
    )

    X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome Variable', axis=1), data['Outcome Variable'], test_size=0.2)

    RNF.fit(X_train, y_train)
    preds = RNF.predict(X_test)
    
    # Classification report and accuracy score
    print(accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
 # ROC-AUC score and plot
    roc_auc = roc_auc_score(y_test, preds)
    y_prob = RNF.predict_proba(X_test)
    roc_auc_plot = skplt.metrics.plot_roc(y_test, y_prob)
    st.pyplot(roc_auc_plot.get_figure())

# Confusion matrix heatmap
    cf_matrix = pd.DataFrame(confusion_matrix(y_test, preds), index=['Real Positive', 'Real Negative'], columns=['Pred Positive', 'Pred Negative'])
    confusion_matrix_heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues_r', cbar=False, linewidth=10)
    st.pyplot(confusion_matrix_heatmap.figure)  # Display the Seaborn plot using Streamlit


    # PCA analysis
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(X_train_pca.shape, y_train.shape)
    print(X_test_pca.shape, y_test.shape)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    
    
    # Linear regression with polynomial features
    X = data.drop('Outcome Variable', axis=1)
    y = data['Outcome Variable']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    
    # Select top k features using ANOVA F-value
    k = 10
    X = data.drop(['Outcome Variable'], axis=1)
    y = data['Outcome Variable']
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    data = data[list(selected_features) + ['Outcome Variable']]
    
    # Return the engineered features
    return data

# Perform feature engineering on the data
engineered_features = perform_feature_engineering(df, feature_engineering_opportunities)


# Step 7: Evaluate the impact
def evaluate_model_performance(features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Fit a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model's performance
    accuracy = model.score(X_test, y_test)

    # Other evaluation metrics
    classification_report_str = classification_report(y_test, model.predict(X_test))
    roc_auc_plot = skplt.metrics.plot_roc(y_test, model.predict_proba(X_test))
    confusion_matrix_heatmap = sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, cmap='Blues_r', cbar=False, linewidth=10)

    # Create a dictionary to hold the evaluation results
    evaluation_results = {
        'accuracy': accuracy,
        'classification_report': classification_report_str,
        'roc_auc_plot': roc_auc_plot,
        'confusion_matrix_heatmap': confusion_matrix_heatmap
    }

    return evaluation_results

# Step 8: Iterate and refine
def iterate_and_refine(data, target):
    # Iterate on the feature engineering process, experimenting with different techniques
    # and combinations of features
    # Refine the feature engineering steps based on model performance and insights

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Perform EDA
    eda_results = perform_eda(preprocessed_data)

    # Identify feature engineering opportunities
    feature_engineering_opportunities = identify_feature_engineering_opportunities(preprocessed_data, target)

    # Perform feature engineering
    engineered_features = perform_feature_engineering(preprocessed_data, feature_engineering_opportunities)

    # Evaluate the impact of feature engineering
    evaluation_results = evaluate_model_performance(engineered_features, target)
    return evaluation_results

# Example usage
prompt = "Consultancy area: Med"
recommended_datasets = get_dataset_recommendations(prompt)
dataset_path = recommended_datasets[0]  # Assuming the first dataset is selected
dataset = load_dataset(dataset_path)
target = dataset['Outcome Variable']  # Use the correct column name here

# Run the feature engineering pipeline
iterate_and_refine(dataset, target)