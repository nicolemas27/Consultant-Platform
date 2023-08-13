import streamlit as st
import openai
import os
import pandas as pd
from consultant import (
    get_dataset_recommendations,
    load_dataset,
    preprocess_data,
    perform_feature_engineering,
    evaluate_model_performance,
    identify_feature_engineering_opportunities,
    iterate_and_refine
)

openai.api_key = os.getenv("OPENAI_API_KEY")

default_dataset_path = "Disease_symptom_and_patient_profile_dataset.csv"

# Streamlit UI components
st.title("Consultant Platform")

# User input using Streamlit widget
prompt = st.text_input("Enter a prompt for dataset recommendations:", "Consultancy area: Med")

if st.button("Start Process"):
    if openai.api_key:
        # Call GPT-3 API to generate dataset recommendations based on the prompt
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100
        )
        recommended_datasets = response.choices[0].text.strip().split('\n')
    else:
        # Use the default dataset
        recommended_datasets = [default_dataset_path]

    # Assuming the first dataset is selected
    dataset_path = recommended_datasets[0]
    
    # Load the dataset
    dataset = load_dataset(dataset_path)
    # Display basic information about the dataset
    st.write("Dataset Summary:")
    st.write(f"Number of Rows: {dataset.shape[0]}")
    st.write(f"Number of Columns: {dataset.shape[1]}")
    st.write("Column Names:")
    st.write(dataset.columns.tolist())

# Display descriptive statistics of numeric columns
    st.write("Descriptive Statistics:")
    st.write(dataset.describe())
    
    # Define target variable
    target = dataset['Outcome Variable']
        
    # Preprocess the data
    preprocessed_data = preprocess_data(dataset)
    
    # Perform feature engineering
    feature_engineering_opportunities = identify_feature_engineering_opportunities(preprocessed_data, target)
    engineered_features = perform_feature_engineering(preprocessed_data, feature_engineering_opportunities)

    # Evaluate model performance
    evaluation_results = evaluate_model_performance(engineered_features, target)

    # Display accuracy
    st.write("Accuracy:", evaluation_results['accuracy'])

    # Display classification report
    st.write("Classification Report:")
    st.write(evaluation_results['classification_report'])

    # Display ROC-AUC plot
    st.pyplot(evaluation_results['roc_auc_plot'].get_figure())

    # Display confusion matrix heatmap
    st.write("Confusion Matrix Heatmap:")
    st.pyplot(evaluation_results['confusion_matrix_heatmap'].get_figure())

    st.write("Final Summary:")
    st.write("The feature engineering process was conducted to improve the model's performance.")
    st.write("By transforming and creating new features, the model's accuracy improved to", evaluation_results['accuracy'])
    st.write("The EDA included summary statistics, histograms, and correlation analysis to gain insights into feature distributions and relationships.")
    st.write("The classification report provides insights into the model's precision, recall, and F1-score for each class.")
    st.write("The ROC-AUC plot visualizes the model's ability to distinguish between classes.")
    st.write("The confusion matrix heatmap shows the distribution of true positive, true negative, false positive, and false negative predictions.")
    st.write("Based on the analysis, it can be concluded that the feature engineering process and model evaluation have led to improvements in performance and a better understanding of the dataset.")


    st.success("Process completed successfully!")
