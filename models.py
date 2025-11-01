from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from datetime import datetime, timezone

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='user')

    datasets = db.relationship('Dataset', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def __repr__(self) -> str:
        return f"<User {self.username}>"
    
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(100), nullable=False, default='Raw')
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone.utc))
    filesize = db.Column(db.Integer, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self) -> str:
        return f"<Dataset {self.filename} owned by User ID {self.user_id}>"
    
    def get_filesize(self):
        if self.filesize < 1024:
            return f"{self.filesize} B"
        elif self.filesize < 1024 ** 2:
            return f"{self.filesize / 1024:.2f} KB"
        elif self.filesize < 1024 ** 3:
            return f"{self.filesize / (1024 ** 2):.2f} MB"
        elif self.filesize < 1024 ** 4:
            return f"{self.filesize / (1024 ** 3):.2f} GB"
        else:
            return f"{self.filesize / (1024 ** 4):.2f} TB"