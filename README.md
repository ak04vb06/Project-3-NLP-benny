# Project-3-NLP-Business-Case-Automated-Customers-Reviews-Week-6-benny

Project Overview
RobotReview is an end-to-end data science project designed to transform thousands of raw customer reviews into structured, actionable buyer's guides. The system utilizes a multi-stage NLP pipeline to categorize products, evaluate consumer sentiment, and identify recurring issues through unsupervised learning.

Technical Architecture
1. Data Preprocessing and Categorization
The dataset consists of over 67,000 Amazon product reviews. The pipeline filters and maps these into six core "Elite" categories:

Tablets (Med 8", Mini 7", Kids Edition)

Streaming and TV Devices

Smart Home and Speakers

Household Batteries

2. Sentiment Analysis (Task 1)
To determine consumer satisfaction, a Sentiment Analysis model was trained using TF-IDF vectorization and Logistic Regression.

Accuracy: Approximately 89% on the test set.

Balanced Training: The model was trained on an oversampled/balanced dataset to ensure high precision across Positive, Neutral, and Negative sentiments.

3. Feature Extraction and Clustering (Task 2)
The system focuses specifically on negative feedback to help potential buyers understand product risks.

Methodology: K-Means clustering is applied to reviews predicted as negative by the sentiment model.

Extraction: The algorithm groups similar complaints into three distinct "Issue Clusters," extracting key descriptors to define the main flaws of each category.

4. Hybrid Content Generation (Task 3)
The project features a dual-engine generation system to compare Cloud-based and Local AI capabilities:

Cloud Engine: Integration with the Google Gemini 2.5 Flash API for high-speed, high-quality copywriting.

Local Engine: Deployment of the Qwen 2.5 (3B-Instruct) model running locally via the Hugging Face Transformers library. This engine leverages the NVIDIA RTX 5070 (6GB VRAM) using CUDA acceleration.

Application Interface
The final output is presented through a Streamlit-based dashboard allowing users to:

Select product categories dynamically.

Toggle between Cloud (Gemini) and Local (RTX 5070) generation.

Visualize the "Verdict" derived from both positive highlights and clustered negative flaws.

Installation and Execution
Prerequisites
Python 3.9+

NVIDIA GPU with CUDA support (for Local Inference)

Gemini API Key (stored in a .env file)

Setup
Install dependencies:

Bash
pip install -r requirements.txt

Generate the data backend:
Run all cells in main.ipynb to process the raw CSV files and generate articles_blog.csv.

Launch the dashboard:

Bash
streamlit run app.py