import os
import logging
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import io
import traceback

# Load environment variables
load_dotenv()

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Railway-specific configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(24).hex()),
    MAX_CONTENT_LENGTH=int(os.environ.get('MAX_UPLOAD_SIZE', 10 * 1024 * 1024)),  # 10MB default
    SESSION_COOKIE_SECURE=True if os.environ.get('RAILWAY_ENVIRONMENT') == 'production' else False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    # Railway provides PORT environment variable
    SERVER_NAME=os.environ.get('RAILWAY_PUBLIC_DOMAIN', None),
)

# File upload configuration
ALLOWED_EXTENSIONS = {'csv', 'txt'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'downloads'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple ML Model for demo (replace with your actual model)
class LoanPredictor:
    def __init__(self):
        self.model = self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a simple one for demo"""
        try:
            # Try to load from Railway's persistent storage
            model_path = os.environ.get('MODEL_PATH', 'model/trained_model.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
                return model
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        
        # Create a simple model for demo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000, 
            n_features=15,
            n_informative=10,
            random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        logger.info("Created demo model")
        return model
    
    def predict(self, data):
        """Make prediction based on input data"""
        try:
            # Convert input to feature array
            features = self.extract_features(data)
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0][1]
            
            return {
                'eligible': bool(prediction),
                'probability': float(probability),
                'confidence': 'High' if probability > 0.8 or probability < 0.2 else 'Medium',
                'risk_level': 'Low' if probability > 0.7 else 'Medium' if probability > 0.4 else 'High'
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def extract_features(self, data):
        """Extract features from input data"""
        # Default feature values
        features = [
            data.get('income', 50000) / 100000,  # Normalized income
            data.get('credit_score', 650) / 850,  # Normalized credit score
            data.get('debt_to_income_ratio', 0.3),
            min(data.get('existing_loans', 0) / 5, 1),
            data.get('employment_years', 5) / 20,
            data.get('age', 35) / 100,
            data.get('savings_balance', 10000) / 100000,
            data.get('loan_amount_requested', 10000) / data.get('income', 50000),
            1 if data.get('education') in ['Master', 'PhD'] else 0,
            1 if data.get('marital_status') == 'Married' else 0,
            1 if data.get('home_ownership') == 'Own' else 0,
            data.get('dependents', 0) / 5,
            min(data.get('late_payments_6m', 0) / 6, 1),
            1 if data.get('bankruptcies', 0) > 0 else 0
        ]
        
        # Pad to expected feature count
        expected_features = 20
        if len(features) < expected_features:
            features += [0] * (expected_features - len(features))
        
        return features[:expected_features]

# Initialize predictor
predictor = LoanPredictor()

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'development'),
        'service': 'loan-predictor'
    }), 200

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['income', 'credit_score', 'debt_to_income_ratio']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Web interface for predictions"""
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = {
                'age': int(request.form.get('age', 35)),
                'income': float(request.form.get('income', 50000)),
                'education': request.form.get('education', 'Bachelor'),
                'employment_years': float(request.form.get('employment_years', 5)),
                'credit_score': int(request.form.get('credit_score', 650)),
                'debt_to_income_ratio': float(request.form.get('debt_to_income_ratio', 0.3)),
                'existing_loans': int(request.form.get('existing_loans', 0)),
                'savings_balance': float(request.form.get('savings_balance', 10000)),
                'mortgage_balance': float(request.form.get('mortgage_balance', 0)),
                'loan_amount_requested': float(request.form.get('loan_amount_requested', 10000)),
                'marital_status': request.form.get('marital_status', 'Single'),
                'dependents': int(request.form.get('dependents', 0)),
                'home_ownership': request.form.get('home_ownership', 'Rent'),
                'late_payments_6m': int(request.form.get('late_payments_6m', 0)),
                'bankruptcies': int(request.form.get('bankruptcies', 0))
            }
            
            # Make prediction
            result = predictor.predict(form_data)
            
            return render_template('results.html', 
                                 result=result, 
                                 form_data=form_data,
                                 timestamp=datetime.now())
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction from CSV upload"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Read CSV
                df = pd.read_csv(file)
                
                # Process each row
                predictions = []
                for _, row in df.iterrows():
                    try:
                        prediction = predictor.predict(row.to_dict())
                        predictions.append({
                            **row.to_dict(),
                            **prediction
                        })
                    except Exception as e:
                        predictions.append({
                            **row.to_dict(),
                            'error': str(e)
                        })
                
                # Convert to DataFrame
                results_df = pd.DataFrame(predictions)
                
                # Create download file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f'predictions_{timestamp}.csv'
                output_path = os.path.join('static', 'downloads', output_filename)
                results_df.to_csv(output_path, index=False)
                
                # Generate summary
                successful = results_df[~results_df['eligible'].isna()] if 'eligible' in results_df.columns else pd.DataFrame()
                
                summary = {
                    'total_records': len(results_df),
                    'processed': len(successful),
                    'failed': len(results_df) - len(successful),
                    'approved': int(successful['eligible'].sum()) if not successful.empty else 0,
                    'approval_rate': (successful['eligible'].mean() * 100) if not successful.empty else 0
                }
                
                return render_template('batch_results.html',
                                     summary=summary,
                                     download_file=output_filename,
                                     sample_data=results_df.head(10).to_dict('records'))
                
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Only CSV files are allowed', 'error')
            return redirect(request.url)
    
    return render_template('batch.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files"""
    try:
        filepath = os.path.join('static', 'downloads', secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        flash('Error downloading file', 'error')
        return redirect(url_for('index'))

@app.route('/api/docs')
def api_docs():
    """API documentation"""
    docs = {
        'endpoints': {
            '/api/predict': {
                'method': 'POST',
                'description': 'Predict loan eligibility',
                'required_fields': ['income', 'credit_score', 'debt_to_income_ratio'],
                'example_request': {
                    'age': 35,
                    'income': 75000,
                    'education': 'Master',
                    'credit_score': 720,
                    'debt_to_income_ratio': 0.3,
                    'loan_amount_requested': 20000
                }
            },
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            }
        }
    }
    return jsonify(docs)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Railway startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting loan predictor on port {port} (debug={debug})")
    logger.info(f"Railway environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'Not set')}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)