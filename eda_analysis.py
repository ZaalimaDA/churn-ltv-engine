import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql("SELECT * FROM customers", engine)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

sns.set_theme(style='whitegrid', palette='muted')
os.makedirs('eda_outputs', exist_ok=True)
