import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from models import db, bcrypt, User, Dataset, Experiment
from werkzeug.utils import secure_filename
from utils import load_dataset, get_data_summary, apply_transformations
from ml_models import AVAILABLE_MODELS, run_linear_regression, run_logistic_regression

app = Flask(__name__)
app.config.from_object(Config)

@app.before_request
def make_session_permanent():
    session.permanent = True

db.init_app(app)
bcrypt.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' #type: ignore
login_manager.login_message_category = 'info'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User()
        new_user.username = username
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/upload-dataset', methods=['POST'])
@login_required
def upload_dataset():
    if 'dataset_file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(request.referrer or url_for('datasets'))

    file = request.files['dataset_file']
    if file.filename == '':
        flash('No selected file.', 'danger')
        return redirect(request.referrer or url_for('datasets'))

    if file and allowed_file(file.filename):
        orig_filename = secure_filename(file.filename) # type: ignore
        ext = orig_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            file.save(save_path)
            filesize = os.path.getsize(save_path)

            new_dataset = Dataset()
            new_dataset.filename = orig_filename
            new_dataset.filepath = unique_filename
            new_dataset.filesize = filesize
            new_dataset.status = 'Raw'
            new_dataset.user_id = current_user.id

            db.session.add(new_dataset)
            db.session.commit()

            flash(f'Dataset "{orig_filename}" uploaded successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while uploading the file: {str(e)}', 'danger')
        return redirect(url_for('datasets'))

    flash('Invalid file type. Please upload a valid dataset file.', 'danger')
    return redirect(request.referrer or url_for('datasets'))

@app.route('/delete-dataset/<int:dataset_id>')
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.owner != current_user:
        flash('You do not have permission to delete this dataset.', 'danger')
        return redirect(url_for('datasets'))

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filepath)
        if os.path.exists(file_path):
            os.remove(file_path)

        db.session.delete(dataset)
        db.session.commit()
        flash(f'Dataset "{dataset.filename}" has been deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the dataset: {str(e)}', 'danger')

    return redirect(url_for('datasets'))

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/datasets')
@login_required
def datasets():
    user_datasets = current_user.datasets
    return render_template('datasets.html', datasets=user_datasets)

@app.route('/preprocess/<int:dataset_id>')
@login_required
def preprocess(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.owner != current_user:
        flash('You do not have permission to access this dataset.', 'danger')
        return redirect(url_for('datasets'))
    
    df, error = load_dataset(dataset)
    if error:
        flash(f"Could not load dataset: {error}", 'danger')
        return redirect(url_for('datasets'))
    
    data_summary = get_data_summary(df)
    return render_template('preprocess.html', dataset=dataset, data_summary=data_summary)

@app.route('/apply-preprocess/<int:dataset_id>', methods=['POST'])
@login_required
def apply_preprocess(dataset_id):
    original_dataset = Dataset.query.get_or_404(dataset_id)
    if original_dataset.owner != current_user:
        flash('You do not have permission to modify this dataset.', 'danger')
        return redirect(url_for('datasets'))

    df, error = load_dataset(original_dataset)
    if error:
        flash(f"Could not load dataset: {error}", 'danger')
        return redirect(url_for('preprocess', dataset_id=dataset_id))

    form_data = request.form
    transformed_df, error = apply_transformations(df, form_data)
    if error:
        flash(f"Could not apply transformations: {error}", 'danger')
        return redirect(url_for('preprocess', dataset_id=dataset_id))

    try:
        new_filename = f"{uuid.uuid4().hex}.csv"
        new_save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        base_name = os.path.splitext(original_dataset.filename)[0]
        new_display_name = f"{base_name}_preprocessed.csv"

        transformed_df.to_csv(new_save_path, index=False)
        new_filesize = os.path.getsize(new_save_path)

        new_dataset = Dataset()
        new_dataset.filename = new_display_name
        new_dataset.filepath = new_filename
        new_dataset.filesize = new_filesize
        new_dataset.status = 'Preprocessed'
        new_dataset.user_id = current_user.id

        db.session.add(new_dataset)
        db.session.commit()

        flash(f'Preprocessing applied successfully! New dataset "{new_display_name}" created.', 'success')
        return redirect(url_for('datasets'))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while applying preprocessing: {str(e)}', 'danger')
        return redirect(url_for('preprocess', dataset_id=dataset_id))
    
@app.route('/models')
@login_required
def models():
    return render_template('models.html')

@app.route('/train')
@login_required
def train():
    datasets = current_user.datasets

    preprocessed_datasets = [ds for ds in datasets if ds.status == 'Preprocessed']
    raw_datasets = [ds for ds in datasets if ds.status == 'Raw']

    return render_template(
        'train.html',
        preprocessed_datasets=preprocessed_datasets,
        raw_datasets=raw_datasets,
        all_models=AVAILABLE_MODELS
    )

@app.route('/start-training', methods=['POST'])
@login_required
def start_training():
    exp_name = request.form.get('exp_name')
    dataset_id = request.form.get('dataset_id')
    target_variable = request.form.get('target_variable')
    model_type = request.form.get('model_type')

    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.owner != current_user:
        flash('You do not have permission to use this dataset.', 'danger')
        return redirect(url_for('train'))
    
    if dataset.status != 'Preprocessed':
        flash('Selected dataset is not suitable for training.', 'danger')
        return redirect(url_for('train'))

    new_experiment = Experiment()
    new_experiment.name = exp_name
    new_experiment.status = 'Running'
    new_experiment.user_id = current_user.id
    new_experiment.dataset_id = dataset.id
    new_experiment.model_type = model_type
    new_experiment.target_variable = target_variable
    new_experiment.parameters = {}

    db.session.add(new_experiment)
    db.session.commit()

    try:
        if model_type == 'linear_regression':
            hyperparameters = {'fit_intercept': request.form.get('lr_fit_intercept') == 'True'}
            new_experiment.parameters = hyperparameters

            model, metrics, error = run_linear_regression(dataset, target_variable, hyperparameters)
            if error:
                raise Exception(error)
            
            new_experiment.status = 'Complete'
            new_experiment.metrics = metrics
            db.session.commit()

            flash(f'Experiment "{exp_name}" completed successfully!', 'success')
            return redirect(url_for('experiments'))
        elif model_type == 'logistic_regression':
            hyperparameters = {
                'penalty': request.form.get('logreg_penalty'),
                'C': request.form.get('logr_c', 1.0),
            }
            new_experiment.parameters = hyperparameters

            model, metrics, error = run_logistic_regression(dataset, target_variable, hyperparameters)

            if error:
                raise Exception(error)
            
            new_experiment.status = 'Complete'
            new_experiment.metrics = metrics
            db.session.commit()

            flash(f'Experiment "{exp_name}" completed successfully!', 'success')
            return redirect(url_for('experiments'))
        else:
            raise Exception('Selected model type is not supported yet.')
    except Exception as e:
        db.session.rollback()
        experiment_to_fail = Experiment.query.get(new_experiment.id)
        if experiment_to_fail:
            experiment_to_fail.status = 'Failed'
            experiment_to_fail.metrics = {'error': str(e)}
            db.session.commit()
        else:
            db.session.rollback()

        flash(f'Experiment "{exp_name}" failed: {str(e)}', 'danger')
        return redirect(url_for('train'))

@app.route('/experiments')
@login_required
def experiments():
    user_experiments = Experiment.query.filter_by(user_id=current_user.id).order_by(Experiment.created_at.desc()).all()
    return render_template('experiments.html', experiments=user_experiments)

@app.route('/experiment/<int:experiment_id>')
@login_required
def experiment_detail(experiment_id):
    experiment = Experiment.query.get_or_404(experiment_id)
    if experiment.owner != current_user:
        flash('You do not have permission to view this experiment.', 'danger')
        return redirect(url_for('experiments'))

    return render_template('experiment-result.html', experiment=experiment)

@app.route('/compare')
@login_required
def compare():
    experiments = Experiment.query.filter_by(user_id=current_user.id, status='Complete').order_by(Experiment.dataset_id).all()
    comparisons = {}
    for exp in experiments:
        if exp.dataset_id not in comparisons:
            comparisons[exp.dataset_id] = {
                'dataset_name': exp.dataset.filename,
                'regression': [],
                'classification': []
            }

        if 'mse' in exp.metrics:
            comparisons[exp.dataset_id]['regression'].append(exp)
        elif 'accuracy' in exp.metrics:
            comparisons[exp.dataset_id]['classification'].append(exp)
    
    for data in comparisons.values():
        data['regression'].sort(key=lambda x: x.metrics.get('r2_score', -float('inf')), reverse=True)
        data['classification'].sort(key=lambda x: x.metrics.get('roc_auc', -float('inf')), reverse=True)

    return render_template('compare.html', comparisons=comparisons)

@app.route('/api/get-dataset-columns/<int:dataset_id>')
@login_required
def get_dataset_columns(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.owner != current_user:
        return jsonify({'error': 'Unauthorized access'}), 403

    df, error = load_dataset(dataset)
    if error:
        return jsonify({'error': f"Could not load dataset: {error}"}), 400

    columns_info = {}
    for col in df.columns: # type: ignore
        if pd.api.types.is_numeric_dtype(df[col]): # type: ignore
            columns_info[col] = 'numeric'
        else:
            columns_info[col] = 'categorical'

    return jsonify(columns_info)

def main():
    with app.app_context():
        db.create_all()
    app.run(debug=True)

main()