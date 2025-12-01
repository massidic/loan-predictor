# app.py - Add these new routes
from flask import send_file
import io

@app.route('/download_template')
def download_template():
    """Download CSV template"""
    try:
        # Generate sample template
        template_df = predictor.generate_sample_template()
        
        # Create in-memory file
        output = io.BytesIO()
        template_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='loan_prediction_template.csv'
        )
        
    except Exception as e:
        logger.error(f"Template download error: {str(e)}")
        flash('Error generating template', 'error')
        return redirect(url_for('batch_predict'))

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch upload via AJAX"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file)
            
            # Make predictions
            results = predictor.predict_batch(df)
            
            # Create download file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"loan_predictions_{timestamp}.csv"
            results.to_csv(f"static/downloads/{output_filename}", index=False)
            
            # Generate summary
            summary = {
                'total_records': len(results),
                'approved_count': results['loan_approved_prediction'].sum(),
                'approval_rate': (results['loan_approved_prediction'].sum() / len(results)) * 100,
                'average_probability': results['approval_probability'].mean() * 100
            }
            
            return jsonify({
                'success': True,
                'summary': summary,
                'download_file': output_filename,
                'sample_data': results.head(10).to_dict('records')
            })
        
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            results = predictor.predict_batch(df)
            
            # Convert to JSON-friendly format
            results_dict = results.to_dict('records')
            
            return jsonify({
                'success': True,
                'predictions': results_dict,
                'total_records': len(results),
                'approved_count': results['loan_approved_prediction'].sum(),
                'timestamp': datetime.now().isoformat()
            })
        
        else:
            return jsonify({'error': 'File must be in CSV format'}), 400
            
    except Exception as e:
        logger.error(f"API batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500