from flask import Flask, render_template, request, jsonify, send_file
from flask.views import MethodView
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Utility function to convert matplotlib figure to base64 image
def fig_to_base64(fig):
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

class VATAnalysisView(MethodView):
    def get(self):
        return render_template('upload.html')
    
    def post(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and file.filename.endswith('.xlsx'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Run analysis
                analysis_results = run_comprehensive_vat_analysis(filepath)
                
                # Convert matplotlib figures to base64 images
                viz_images = {}
                for key, fig in analysis_results['visualizations'].items():
                    if fig is not None:
                        viz_images[key] = fig_to_base64(fig)
                
                # Prepare data for the template
                template_data = {
                    'report': analysis_results['report'],
                    'viz_images': viz_images,
                    'sales_summary': analysis_results['data']['sales_df'].head(10).to_html(classes='table table-striped'),
                    'vat_summary': analysis_results['data']['vat_df'].to_html(classes='table table-striped'),
                    'total_summary': analysis_results['data']['total_df'].to_html(classes='table table-striped'),
                    'model_performance': {
                        name: f"{info['score']:.4f}" 
                        for name, info in analysis_results['analyses']['predictive']['models'].items()
                    },
                    'future_predictions': {
                        model: [f"{val:.2f}" for val in values]
                        for model, values in analysis_results['analyses']['predictive']['future_predictions'].items()
                    },
                    'cluster_profiles': analysis_results['analyses']['unsupervised']['cluster_profiles'].to_html(classes='table table-striped'),
                    'high_efficiency': analysis_results['analyses']['prescriptive']['high_efficiency_categories'].to_html(classes='table table-striped'),
                    'low_efficiency': analysis_results['analyses']['prescriptive']['low_efficiency_categories'].to_html(classes='table table-striped'),
                    'scenarios': {
                        scenario: f"{value:.2f}" 
                        for scenario, value in analysis_results['analyses']['prescriptive']['scenarios'].items()
                    }
                }
                
                return render_template('results.html', **template_data)
                
            except Exception as e:
                return jsonify({'error': f'Analysis failed: {str(e)}'})
        
        return jsonify({'error': 'Invalid file format. Please upload an Excel file (.xlsx)'})

# Register the view
app.add_url_rule('/vat-analysis', view_func=VATAnalysisView.as_view('vat_analysis'))

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Create upload.html template
with open('templates/upload.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VAT Computation Analysis</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 50px;
        }
        .upload-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VAT Computation Analysis System</h1>
            <p class="lead">Upload your VAT computation Excel file to analyze and optimize your VAT claims</p>
        </div>
        
        <div class="upload-container">
            <form action="/vat-analysis" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Select Excel File (.xlsx)</label>
                    <input type="file" class="form-control-file" id="file" name="file" required accept=".xlsx">
                </div>
                <div class="form-group">
                    <p class="text-muted">The file should contain VAT computation data with sales categories, VAT calculations, and claimable amounts.</p>
                </div>
                <button type="submit" class="btn btn-primary btn-block">Analyze</button>
            </form>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
    """)

# Create results.html template
with open('templates/results.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VAT Analysis Results</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 40px;
        }
        .viz-container {
            margin-bottom: 40px;
        }
        .viz-img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .report-section {
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .nav-pills .nav-link.active {
            background-color: #007bff;
        }
        .tab-content {
            padding: 20px;
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">VAT Computation Analysis Results</h1>
        
        <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
            <li class="nav-item" role="presentation">
                <a class="nav-link active" id="pills-summary-tab" data-toggle="pill" href="#pills-summary" role="tab">Executive Summary</a>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="pills-descriptive-tab" data-toggle="pill" href="#pills-descriptive" role="tab">Descriptive Analysis</a>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="pills-predictive-tab" data-toggle="pill" href="#pills-predictive" role="tab">Predictive Analysis</a>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="pills-prescriptive-tab" data-toggle="pill" href="#pills-prescriptive" role="tab">Prescriptive Analysis</a>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="pills-unsupervised-tab" data-toggle="pill" href="#pills-unsupervised" role="tab">Pattern Discovery</a>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" id="pills-report-tab" data-toggle="pill" href="#pills-report" role="tab">Full Report</a>
            </li>
        </ul>
        
        <div class="tab-content" id="pills-tabContent">
            <!-- Executive Summary Tab -->
            <div class="tab-pane fade show active" id="pills-summary" role="tabpanel">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">VAT Summary</h4>
                            </div>
                            <div class="card-body">
                                {{ vat_summary|safe }}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Sales Summary</h4>
                            </div>
                            <div class="card-body">
                                {{ total_summary|safe }}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Key Optimization Scenarios</h4>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Scenario</th>
                                        <th>Claimable Amount</th>
                                        <th>Improvement</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for scenario, value in scenarios.items() %}
                                    <tr>
                                        <td>{{ scenario }}</td>
                                        <td>{{ value }}</td>
                                        <td>
                                            {% if scenario != 'Current' %}
                                                {{ ((float(value) - float(scenarios['Current'])) / float(scenarios['Current']) * 100)|round(2) }}%
                                            {% else %}
                                                -
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Descriptive Analysis Tab -->
            <div class="tab-pane fade" id="pills-descriptive" role="tabpanel">
                <h3 class="mb-4">Sales and VAT Distribution</h3>
                
                <div class="row">
                    <div class="col-md-12 mb-4">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Descriptive Visualization</h4>
                            </div>
                            <div class="card-body text-center">
                                {% if viz_images.descriptive %}
                                    <img src="data:image/png;base64,{{ viz_images.descriptive }}" class="viz-img">
                                {% else %}
                                    <p class="text-muted">Visualization not available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <h3 class="mb-4">Top Sales Categories</h3>
                <div class="table-responsive">
                    {{ sales_summary|safe }}
                </div>
            </div>
            
            <!-- Predictive Analysis Tab -->
            <div class="tab-pane fade" id="pills-predictive" role="tabpanel">
                <h3 class="mb-4">Sales Forecasting</h3>
                
                <div class="row">
                    <div class="col-md-12 mb-4">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Predictive Visualization</h4>
                            </div>
                            <div class="card-body text-center">
                                {% if viz_images.predictive %}
                                    <img src="data:image/png;base64,{{ viz_images.predictive }}" class="viz-img">
                                {% else %}
                                    <p class="text-muted">Visualization not available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Model Performance</h4>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>R² Score</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for model, score in model_performance.items() %}
                                            <tr>
                                                <td>{{ model }}</td>
                                                <td>{{ score }}</td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Future Predictions (Next 3 Months)</h4>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Month 1</th>
                                                <th>Month 2</th>
                                                <th>Month 3</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for model, values in future_predictions.items() %}
                                            <tr>
                                                <td>{{ model }}</td>
                                                {% for val in values %}
                                                <td>{{ val }}</td>
                                                {% endfor %}
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Prescriptive Analysis Tab -->
            <div class="tab-pane fade" id="pills-prescriptive" role="tabpanel">
                <h3 class="mb-4">VAT Claim Optimization</h3>
                
                <div class="row">
                    <div class="col-md-12 mb-4">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Prescriptive Visualization</h4>
                            </div>
                            <div class="card-body text-center">
                                {% if viz_images.prescriptive %}
                                    <img src="data:image/png;base64,{{ viz_images.prescriptive }}" class="viz-img">
                                {% else %}
                                    <p class="text-muted">Visualization not available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-info text-white">
                                <h4 class="mb-0">High Efficiency Categories</h4>
                            </div>
                            <div class="card-body">
                                {{ high_efficiency|safe }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card mb-4">
                            <div class="card-header bg-warning text-dark">
                                <h4 class="mb-0">Low Efficiency Categories</h4>
                            </div>
                            <div class="card-body">
                                {{ low_efficiency|safe }}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Unsupervised Analysis Tab -->
            <div class="tab-pane fade" id="pills-unsupervised" role="tabpanel">
                <h3 class="mb-4">Pattern Discovery</h3>
                
                <div class="row">
                    <div class="col-md-12 mb-4">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Cluster Analysis Visualization</h4>
                            </div>
                            <div class="card-body text-center">
                                {% if viz_images.unsupervised %}
                                    <img src="data:image/png;base64,{{ viz_images.unsupervised }}" class="viz-img">
                                {% else %}
                                    <p class="text-muted">Visualization not available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Cluster Profiles</h4>
                    </div>
                    <div class="card-body">
                        {{ cluster_profiles|safe }}
                    </div>
                </div>
            </div>
            
            <!-- Full Report Tab -->
            <div class="tab-pane fade" id="pills-report" role="tabpanel">
                <div class="report-section">
                    <pre style="white-space: pre-wrap;">{{ report }}</pre>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
    """)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True, port=5000)
