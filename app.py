import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from models import db, bcrypt, User, Dataset
from werkzeug.utils import secure_filename
from utils import load_dataset, get_data_summary, apply_transformations

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
        new_filename = f"{uuid.uuid4().hex}.{original_dataset.filepath.rsplit('.', 1)[1]}"
        new_save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        base_name = os.path.splitext(original_dataset.filename)[0]
        new_display_name = f"{base_name}_preprocessed.{original_dataset.filename.rsplit('.', 1)[1]}"

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

    return render_template('train.html', preprocessed_datasets=preprocessed_datasets, raw_datasets=raw_datasets)

@app.route('/experiments')
@login_required
def experiments():
    return render_template('experiments.html')

@app.route('/experiment/<int:experiment_id>')
@login_required
def experiment_detail(experiment_id):
    return render_template('experiment_detail.html', experiment_id=experiment_id)

@app.route('/compare')
@login_required
def compare():
    return render_template('compare.html')

def main():
    with app.app_context():
        db.create_all()
    app.run(debug=True)

main()