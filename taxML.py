import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
import re
import time
import pickle
import os
from flask import Flask, request, render_template, jsonify

# Load data from Excel file
df = pd.read_excel('Input16%.xlsx')

# Display dataset information
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Preprocessing functions
def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text):
    """Extract numerical features from text"""
    if not isinstance(text, str):
        return 0
    # Extract numbers
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return 0

# Apply preprocessing
df['Processed_Description'] = df['Description of Goods/Services'].apply(preprocess_text)
df['Numerical_Feature'] = df['Description of Goods/Services'].apply(extract_features)

# Encode the target variable (custom entry numbers)
label_encoder = LabelEncoder()
df['Encoded_Entry'] = label_encoder.fit_transform(df['Custom Entry Number'])
print(f"Classes: {label_encoder.classes_}")

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Description'])

# Feature extraction using Bag of Words
count_vectorizer = CountVectorizer(max_features=100)
X_bow = count_vectorizer.fit_transform(df['Processed_Description'])

# Combine with numerical feature
X_tfidf_combined = np.hstack((X_tfidf.toarray(), df['Numerical_Feature'].values.reshape(-1, 1)))
X_bow_combined = np.hstack((X_bow.toarray(), df['Numerical_Feature'].values.reshape(-1, 1)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf_combined, df['Encoded_Entry'], test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Function to evaluate supervised models
def evaluate_supervised_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Round predictions to nearest integer for regression models
    if isinstance(model, (LinearRegression, Ridge, Lasso, RandomForestRegressor, 
                          GradientBoostingRegressor, SVR, KNeighborsRegressor, MLPRegressor)):
        y_pred = np.round(y_pred).astype(int)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (treating as classification problem)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    return {
        'Model': model_name,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Accuracy': accuracy,
        'Training Time': training_time
    }

# Function to evaluate unsupervised models
def evaluate_unsupervised_model(model, X, y, model_name, method_type):
    start_time = time.time()
    
    if method_type == 'clustering':
        # For clustering methods
        clusters = model.fit_predict(X)
        
        # Measure how well clusters separate different classes
        # Using adjusted rand index
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        ari = adjusted_rand_score(y, clusters)
        
        # Silhouette score
        try:
            silhouette = silhouette_score(X, clusters)
        except:
            silhouette = 0  # Some clustering methods might fail for silhouette
            
        performance = {
            'Model': model_name,
            'ARI': ari,
            'Silhouette': silhouette,
            'Training Time': time.time() - start_time
        }
        
    else:  # dimensionality reduction
        # For dimensionality reduction methods
        # Reduce to 2 dimensions
        if hasattr(model, 'transform'):
            X_reduced = model.fit_transform(X)
        else:
            model.fit(X)
            X_reduced = model.transform(X)
            
        # Train a simple classifier on reduced data
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42
        )
        knn.fit(X_train_red, y_train_red)
        accuracy = knn.score(X_test_red, y_test_red)
        
        # Explained variance (for methods that support it)
        explained_var = 0
        if hasattr(model, 'explained_variance_ratio_'):
            explained_var = sum(model.explained_variance_ratio_)
            
        performance = {
            'Model': model_name,
            'Downstream Accuracy': accuracy,
            'Explained Variance': explained_var,
            'Training Time': time.time() - start_time
        }
        
    return performance

# Initialize supervised models
supervised_models = [
    (LinearRegression(), "Linear Regression"),
    (Ridge(alpha=1.0), "Ridge Regression"),
    (Lasso(alpha=0.1), "Lasso Regression"),
    (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"),
    (GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting"),
    # (SVR(kernel='linear'), "Support Vector Regression"),
    # (KNeighborsRegressor(n_neighbors=5), "K-Nearest Neighbors"),
    # (MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "Neural Network")
]

# Initialize unsupervised models
clustering_models = [
    (KMeans(n_clusters=len(label_encoder.classes_), random_state=42), "K-Means", "clustering"),
    (DBSCAN(eps=3, min_samples=2), "DBSCAN", "clustering"),
    (AgglomerativeClustering(n_clusters=len(label_encoder.classes_)), "Agglomerative Clustering", "clustering"),
    (SpectralClustering(n_clusters=len(label_encoder.classes_), random_state=42), "Spectral Clustering", "clustering"),
    (Birch(n_clusters=len(label_encoder.classes_)), "Birch", "clustering")
]

dim_reduction_models = [
    (PCA(n_components=2), "PCA", "dim_reduction"),
    (TruncatedSVD(n_components=2, random_state=42), "Truncated SVD", "dim_reduction"),
    (NMF(n_components=2, init='random', random_state=42), "NMF", "dim_reduction"),
    (LatentDirichletAllocation(n_components=2, random_state=42), "LDA", "dim_reduction"),
    (TSNE(n_components=2, random_state=42), "t-SNE", "dim_reduction")
]

# Create a directory to save models
os.makedirs('models', exist_ok=True)

# Evaluate supervised models
supervised_results = []
for model, name in supervised_models:
    try:
        result = evaluate_supervised_model(model, X_train, X_test, y_train, y_test, name)
        supervised_results.append(result)
        print(f"Completed evaluation of {name}")
        
        # Save the trained model
        with open(f'models/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Evaluate unsupervised models
unsupervised_results = []

# Clustering models
for model, name, method_type in clustering_models:
    try:
        result = evaluate_unsupervised_model(model, X_train, y_train, name, method_type)
        unsupervised_results.append(result)
        print(f"Completed evaluation of {name}")
        
        # Save the trained model
        with open(f'models/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Dimensionality reduction models
for model, name, method_type in dim_reduction_models:
    try:
        result = evaluate_unsupervised_model(model, X_train.toarray() if hasattr(X_train, 'toarray') else X_train, 
                                            y_train, name, method_type)
        unsupervised_results.append(result)
        print(f"Completed evaluation of {name}")
        
        # Save the trained model
        with open(f'models/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Save vectorizers and label encoder
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
    
with open('models/count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)
    
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Convert results to DataFrames
supervised_df = pd.DataFrame(supervised_results)
supervised_df = supervised_df.sort_values(by='Accuracy', ascending=False)

# Separate clustering and dim reduction results
clustering_results = [r for r in unsupervised_results if 'ARI' in r]
dim_reduction_results = [r for r in unsupervised_results if 'Downstream Accuracy' in r]

clustering_df = pd.DataFrame(clustering_results)
if not clustering_df.empty:
    clustering_df = clustering_df.sort_values(by='ARI', ascending=False)

dim_reduction_df = pd.DataFrame(dim_reduction_results)
if not dim_reduction_df.empty:
    dim_reduction_df = dim_reduction_df.sort_values(by='Downstream Accuracy', ascending=False)

# Display results
print("\n==== Supervised Model Performance (Ranked) ====")
print(supervised_df)

print("\n==== Clustering Model Performance (Ranked) ====")
if not clustering_df.empty:
    print(clustering_df)
else:
    print("No clustering results available")

print("\n==== Dimensionality Reduction Model Performance (Ranked) ====")
if not dim_reduction_df.empty:
    print(dim_reduction_df)
else:
    print("No dimensionality reduction results available")

# Visualize results
plt.figure(figsize=(12, 8))

# Supervised models accuracy
plt.subplot(2, 2, 1)
sns.barplot(x='Accuracy', y='Model', data=supervised_df)
plt.title('Supervised Models - Accuracy')
plt.tight_layout()

# Supervised models R2
plt.subplot(2, 2, 2)
sns.barplot(x='R2', y='Model', data=supervised_df)
plt.title('Supervised Models - R2 Score')
plt.tight_layout()

# Clustering models ARI
if not clustering_df.empty:
    plt.subplot(2, 2, 3)
    sns.barplot(x='ARI', y='Model', data=clustering_df)
    plt.title('Clustering Models - Adjusted Rand Index')
    plt.tight_layout()

# Dimensionality reduction models accuracy
if not dim_reduction_df.empty:
    plt.subplot(2, 2, 4)
    sns.barplot(x='Downstream Accuracy', y='Model', data=dim_reduction_df)
    plt.title('Dimensionality Reduction - Downstream Accuracy')
    plt.tight_layout()

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Function to predict Custom Entry Number from new description
def predict_custom_entry(description, model, vectorizer, label_encoder):
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Extract numerical feature
    numerical_feature = extract_features(description)
    
    # Vectorize the description
    desc_vector = vectorizer.transform([processed_desc])
    
    # Combine with numerical feature
    X_combined = np.hstack((desc_vector.toarray(), np.array([numerical_feature]).reshape(-1, 1)))
    
    # Make prediction
    encoded_prediction = model.predict(X_combined)[0]
    
    # Round prediction for regression models
    if isinstance(model, (LinearRegression, Ridge, Lasso, RandomForestRegressor, 
                          GradientBoostingRegressor)):#, SVR, KNeighborsRegressor, MLPRegressor)):
        encoded_prediction = round(encoded_prediction)
    
    # Ensure prediction is within valid range
    encoded_prediction = max(0, min(encoded_prediction, len(label_encoder.classes_) - 1))
    
    # Convert back to Custom Entry Number
    custom_entry = label_encoder.classes_[encoded_prediction]
    return custom_entry

# Get the best supervised model
if not supervised_df.empty:
    best_model_name = supervised_df.iloc[0]['Model']
    best_model = next(model for model, name in supervised_models if name == best_model_name)
    
    # Example prediction
    sample_description = "TRANSPORT OF FINISHED GOODS FROM THIKA TO MSA BY KBJ 477W-6/12/2024"
    predicted_entry = predict_custom_entry(
        sample_description, 
        best_model, 
        tfidf_vectorizer, 
        label_encoder
    )
    
    print(f"\nExample prediction:")
    print(f"Description: {sample_description}")
    print(f"Predicted Custom Entry Number: {predicted_entry}")

# Summary of findings
print("\n==== Summary of Findings ====")
print(f"Best Supervised Model: {supervised_df.iloc[0]['Model']} with Accuracy: {supervised_df.iloc[0]['Accuracy']:.4f}")

if not clustering_df.empty:
    print(f"Best Clustering Model: {clustering_df.iloc[0]['Model']} with ARI: {clustering_df.iloc[0]['ARI']:.4f}")

if not dim_reduction_df.empty:
    print(f"Best Dimensionality Reduction Model: {dim_reduction_df.iloc[0]['Model']} with Downstream Accuracy: {dim_reduction_df.iloc[0]['Downstream Accuracy']:.4f}")

# Create a Flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_results = None
    if request.method == 'POST':
        description = request.form['description']
        prediction_results = []
        
        # Load all supervised models and make predictions
        for model_name, display_name in supervised_models:
            try:
                model_path = f'models/{display_name.replace(" ", "_").lower()}.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                        vectorizer = pickle.load(f)
                        
                    with open('models/label_encoder.pkl', 'rb') as f:
                        le = pickle.load(f)
                    
                    predicted_class = predict_custom_entry(description, model, vectorizer, le)
                    prediction_results.append({
                        'model_name': display_name,
                        'prediction': predicted_class,
                        'model_type': 'Supervised'
                    })
                    
            except Exception as e:
                print(f"Error predicting with {display_name}: {e}")
        
        # Sort by model type for better display
        prediction_results.sort(key=lambda x: x['model_name'])
    
    return render_template('index.html', prediction_results=prediction_results)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.json
    description = data.get('description', '')
    
    results = []
    
    # Load all supervised models and make predictions
    for model_name, display_name in supervised_models:
        try:
            model_path = f'models/{display_name.replace(" ", "_").lower()}.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
                    
                with open('models/label_encoder.pkl', 'rb') as f:
                    le = pickle.load(f)
                
                predicted_class = predict_custom_entry(description, model, vectorizer, le)
                results.append({
                    'model_name': display_name,
                    'prediction': predicted_class,
                    'model_type': 'Supervised'
                })
                
        except Exception as e:
            print(f"Error predicting with {display_name}: {e}")
    
    return jsonify({'results': results})

# HTML template for the web application
@app.route('/get_template')
def get_template():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Custom Entry Number Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .form-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .info {
            margin-top: 20px;
            padding: 10px;
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
        }
    </style>
</head>
<body>
    <h1>Custom Entry Number Prediction</h1>
    
    <div class="container">
        <div class="form-container">
            <form method="POST">
                <label for="description">Description of Goods/Services:</label>
                <input type="text" id="description" name="description" required placeholder="Enter description here...">
                <button type="submit">Predict</button>
            </form>
            
            <div class="info">
                <p><strong>Example descriptions:</strong></p>
                <ul>
                    <li>LC 1737-PURCHASE OF RICE CHAMANAN LAL SETIA Based On Landed Costs 1737</li>
                    <li>LOCAL TRANSPORTATION FOR RUSSIAN WHEAT</li>
                    <li>TRANSPORT OF FINISHED GOODS FROM THIKA TO MSA BY KBJ 477W-6/12/2024</li>
                </ul>
            </div>
        </div>
        
        {% if prediction_results %}
        <div class="results-container">
            <h2>Prediction Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Predicted Custom Entry Number</th>
                        <th>Model Type</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in prediction_results %}
                    <tr>
                        <td>{{ result.model_name }}</td>
                        <td>{{ result.prediction }}</td>
                        <td>{{ result.model_type }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
    """
    return html

# Create the templates directory
os.makedirs('templates', exist_ok=True)

# Write the template
with open('templates/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Custom Entry Number Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .form-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .info {
            margin-top: 20px;
            padding: 10px;
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
        }
    </style>
</head>
<body>
    <h1>Custom Entry Number Prediction</h1>
    
    <div class="container">
        <div class="form-container">
            <form method="POST">
                <label for="description">Description of Goods/Services:</label>
                <input type="text" id="description" name="description" required placeholder="Enter description here...">
                <button type="submit">Predict</button>
            </form>
            
            <div class="info">
                <p><strong>Example descriptions:</strong></p>
                <ul>
                    <li>LC 1737-PURCHASE OF RICE CHAMANAN LAL SETIA Based On Landed Costs 1737</li>
                    <li>LOCAL TRANSPORTATION FOR RUSSIAN WHEAT</li>
                    <li>TRANSPORT OF FINISHED GOODS FROM THIKA TO MSA BY KBJ 477W-6/12/2024</li>
                </ul>
            </div>
        </div>
        
        {% if prediction_results %}
        <div class="results-container">
            <h2>Prediction Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Predicted Custom Entry Number</th>
                        <th>Model Type</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in prediction_results %}
                    <tr>
                        <td>{{ result.model_name }}</td>
                        <td>{{ result.prediction }}</td>
                        <td>{{ result.model_type }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
""")

if __name__ == "__main__":
    print("\nStarting Flask application...")
    print("Access the web interface at http://127.0.0.1:5000/")
    app.run(debug=True)