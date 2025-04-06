from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from models import db, User
import pandas as pd
import joblib
import os
import json
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
from sklearn.metrics import classification_report

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db.init_app(app)

#Load your custom ensemble model
model = joblib.load('soft_voting_clf.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# Init DB
@app.before_request
def create_tables():
    db.create_all()
    
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful!')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials!")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Assume classification report is precomputed as HTML string (optional)
    # Or use dummy content or fetch from session/db as needed
    with open('static/classification_report.json') as f:
        report_data = json.load(f)

    return render_template(
        'dashboard.html',
        report=report_data
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    uploaded_data = None
    prediction_data = None
    report = None
    accuracy = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            data = pd.read_csv(filepath)

            if 'Attack Type' in data.columns:
                y_true = data['Attack Type']
                X = data.drop(columns=['Attack Type'])
            else:
                X = data
                y_true = None

            preprocessed_data = preprocess_data(data)
            y_true = preprocessed_data['Attack Type']
            X_preprocessed = preprocessed_data.drop(columns=['Attack Type'])

            y_pred = predict_labels(X_preprocessed, y_true)

            if y_true is not None:
                report = classification_report(
                    y_true,
                    label_encoder.inverse_transform(y_pred),
                    output_dict=True
                )
                accuracy = accuracy_score(
                    y_true, label_encoder.inverse_transform(y_pred)
                )
            drop_columns = ['Attack Number']
            uploaded_data = X.drop(columns=drop_columns, errors='ignore')
            
            results = pd.DataFrame(X)
            # results['Prediction'] = label_encoder.inverse_transform(y_pred)
            results['Prediction'] = label_encoder.inverse_transform(results['Attack Number'])
            results = results.drop(columns='Attack Number')

            uploaded_data = results.drop(columns='Prediction').to_dict(orient="records")
            prediction_data = results.to_dict(orient="records")
            
            print(prediction_data)

    return render_template('upload.html',
                           uploaded_data=uploaded_data,
                           prediction_data=prediction_data)


def predict_labels(X, y):
    # Step 1: Get predicted probabilities from all classifiers
    classifier_probs_dict = {}
    for name, clf in model.named_estimators_.items():
        if hasattr(clf, "predict_proba"):
            classifier_probs_dict[name] = clf.predict_proba(X)

    # Step 2: Flatten probs for similarity computation
    classifier_probs = np.array([probs.flatten() for probs in classifier_probs_dict.values()])
    classifier_names = list(classifier_probs_dict.keys())

    # Step 3: Similarity and clustering
    similarity_matrix = cosine_similarity(classifier_probs)
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = linkage(distance_matrix, method='ward')
    threshold = 0.05
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Step 4: Group classifiers by cluster
    cluster_dict = defaultdict(list)
    for name, cluster_id in zip(classifier_names, clusters):
        cluster_dict[cluster_id].append(name)

    # Step 5: Compute cluster probabilities
    cluster_probs = {}
    for cluster_id, classifiers in cluster_dict.items():
        cluster_sum = np.zeros_like(next(iter(classifier_probs_dict.values())))
        for clf_name in classifiers:
            cluster_sum += classifier_probs_dict[clf_name]
        cluster_probs[cluster_id] = cluster_sum / len(classifiers)

    # Step 6: Compute classifier accuracies
    classifier_accuracies = {
        name: accuracy_score(label_encoder.transform(y), np.argmax(probs, axis=1))
        for name, probs in classifier_probs_dict.items()
    }

    # Step 7: Compute and normalize cluster weights
    cluster_weights = {}
    for cluster_id, classifiers in cluster_dict.items():
        acc_sum = sum(classifier_accuracies[clf] for clf in classifiers)
        cluster_weights[cluster_id] = acc_sum / len(classifiers)
        
    print(cluster_weights)
    
    total_weight = sum(cluster_weights.values())
    print(total_weight)
    total_weight = sum(cluster_weights.values())

    if total_weight == 0:
        # Avoid division by zero, fallback to equal weighting or log it
        cluster_weights = {k: 1 / len(cluster_weights) for k in cluster_weights}
    else:
        cluster_weights = {k: v / total_weight for k, v in cluster_weights.items()}
        # cluster_weights = {k: v / total_weight for k, v in cluster_weights.items()}

    # Step 8: Compute final weighted probabilities
    weighted_probs = np.zeros_like(next(iter(cluster_probs.values())))
    for cluster_id, weight in cluster_weights.items():
        weighted_probs += weight * cluster_probs[cluster_id]

    # Step 9: Make final predictions
    final_predictions = np.argmax(weighted_probs, axis=1)

    return final_predictions

def preprocess_data(data):
    # Strip column names
    data.columns = data.columns.str.strip()
    
    print(data.shape)

    # Checking for infinity values
    numeric_cols = data.select_dtypes(include=np.number).columns
    inf_count = np.isinf(data[numeric_cols]).sum()

    # Replacing any infinite values (positive or negative) with NaN (not a number)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculating medians for specific columns
    med_flow_bytes = data['Flow Bytes/s'].median()
    med_flow_packets = data['Flow Packets/s'].median()

    # Filling missing values in specific columns with their respective medians
    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(med_flow_bytes)
    data['Flow Packets/s'] = data['Flow Packets/s'].fillna(med_flow_packets)
        
    print(data.shape)

    
    # Dropping columns with only one unique value
    drop_columns = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk',
       'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
       'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
    ]
    data = data.drop(columns=drop_columns, errors='ignore')

    # Optimize memory
    # for col in data.columns:
    #     col_type = data[col].dtype
    #     if col_type != object:
    #         c_min = data[col].min()
    #         c_max = data[col].max()
    #         if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
    #             data[col] = data[col].astype(np.float32)
    #         elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
    #             data[col] = data[col].astype(np.int32)
                
    print(data.shape)

    # Scale and reduce dimensions
    attacks = data['Attack Type']
    features = data.drop('Attack Type', axis=1)

    scaler = joblib.load('scaler.joblib')
    scaled_features = scaler.fit_transform(features)
    
    print()

    size = len(features.columns) // 2
    ipca = joblib.load('pca.joblib')
    transformed_features = ipca.transform(scaled_features)
    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i+1}' for i in range(size)])
    new_data['Attack Type'] = attacks.values

    return new_data

if __name__ == '__main__':
    app.run(debug=True)