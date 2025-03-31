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

# Let's assume we have data loaded from an Excel file
# Since we don't have actual data, I'll create a synthetic dataset for demonstration
np.random.seed(42)


    df = pd.read_excel('./Output 16%.xlsx')  

# Display dataset information
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Preprocessing functions
def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_features(text):
    """Extract numerical features from text"""
    # Extract numbers
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return 0

# Apply preprocessing
df['Processed_Description'] = df['Description_of_Goods'].apply(preprocess_text)
df['Numerical_Feature'] = df['Description_of_Goods'].apply(extract_features)

# Encode the target variable (custom entry numbers)
label_encoder = LabelEncoder()
df['Encoded_Entry'] = label_encoder.fit_transform(df['Custom_Entry_Number'])

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
    (KNeighborsRegressor(n_neighbors=5), "K-Nearest Neighbors"),
    (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"),
    (GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting"),
    #(SVR(kernel='linear'), "Support Vector Regression"),
    #(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "Neural Network")
]

# Initialize unsupervised models
clustering_models = [
    (KMeans(n_clusters=len(label_encoder.classes_), random_state=42), "K-Means", "clustering"),
    (DBSCAN(eps=3, min_samples=5), "DBSCAN", "clustering"),
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

# Evaluate supervised models
supervised_results = []
for model, name in supervised_models:
    try:
        result = evaluate_supervised_model(model, X_train, X_test, y_train, y_test, name)
        supervised_results.append(result)
        print(f"Completed evaluation of {name}")
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
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Dimensionality reduction models
for model, name, method_type in dim_reduction_models:
    try:
        result = evaluate_unsupervised_model(model, X_train.toarray() if hasattr(X_train, 'toarray') else X_train, 
                                            y_train, name, method_type)
        unsupervised_results.append(result)
        print(f"Completed evaluation of {name}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

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
plt.show()

# Function to predict Custom Entry Number from new description
def predict_custom_entry(description, best_model, vectorizer, label_encoder):
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Extract numerical feature
    numerical_feature = extract_features(description)
    
    # Vectorize the description
    desc_vector = vectorizer.transform([processed_desc])
    
    # Combine with numerical feature
    X_combined = np.hstack((desc_vector.toarray(), np.array([numerical_feature]).reshape(-1, 1)))
    
    # Make prediction
    encoded_prediction = best_model.predict(X_combined)[0]
    
    # Round prediction for regression models
    if isinstance(best_model, (LinearRegression, Ridge, Lasso, RandomForestRegressor, 
                              GradientBoostingRegressor))#, SVR, KNeighborsRegressor, MLPRegressor)):
        encoded_prediction = round(encoded_prediction)
    
    # Convert back to Custom Entry Number
    try:
        custom_entry = label_encoder.inverse_transform([encoded_prediction])[0]
        return custom_entry
    except:
        return "Prediction error: encoded value out of range"

# Get the best supervised model
if not supervised_df.empty:
    best_model_name = supervised_df.iloc[0]['Model']
    best_model = next(model for model, name in supervised_models if name == best_model_name)
    
    # Example prediction
    sample_description = "Laptops - 250 units"
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