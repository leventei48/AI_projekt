# utils/ml_helpers.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from utils.logger import get_logger
from tqdm import tqdm

log = get_logger(__name__)

# Globális változók a feature encoding-hoz
_feature_encoder = None
_cat_cols = None
_num_cols = None

def encode_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Kategorikus oszlopok one‑hot encoding-ja, numerikusok standardizálása.
    DateTime oszlopok kivonása az encodingból.
    """
    global _feature_encoder, _cat_cols, _num_cols
    
    # DateTime oszlopok azonosítása és kivonása
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    df_without_datetime = df.drop(columns=datetime_cols)
    
    # Osztályozzuk az oszlopokat (datetime nélkül)
    if _cat_cols is None or fit:
        _cat_cols = df_without_datetime.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        _num_cols = [c for c in df_without_datetime.columns if c not in _cat_cols]
        
        # Szűrjük ki a 'label_majority' oszlopot, ha van
        if 'label_majority' in _cat_cols:
            _cat_cols.remove('label_majority')
        if 'label_majority' in _num_cols:
            _num_cols.remove('label_majority')
            
        log.info(f"Kategórikus oszlopok ({len(_cat_cols)}): {_cat_cols}")
        log.info(f"Numerikus oszlopok ({len(_num_cols)}): {_num_cols[:5]}...")
        log.info(f"DateTime oszlopok (kihagyva): {datetime_cols}")
    
    # ColumnTransformer definiálása
    transformers = []
    
    if _cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", 
                                 sparse_output=False,
                                 max_categories=200), _cat_cols)
        )
    
    if _num_cols:
        transformers.append(
            ("num", StandardScaler(), _num_cols)
        )
    
    # Ha nincsenek oszlopok, térjünk vissza a df-fel
    if not transformers:
        return df_without_datetime.copy()
    
    if fit or _feature_encoder is None:
        _feature_encoder = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )
        log.info("Feature encoder létrehozva/fitelve.")
    
    # Fit/transform with progress bar
    if fit:
        log.info("Feature encoding (fit) folyamatban...")
        X_enc = _feature_encoder.fit_transform(df_without_datetime)
        feature_names = _feature_encoder.get_feature_names_out()
        encoded_df = pd.DataFrame(X_enc, columns=feature_names, index=df.index)
        log.info(f"Encoded mátrix mérete: {encoded_df.shape}")
        return encoded_df
    else:
        if _feature_encoder is None:
            raise RuntimeError("Encoder még nincs fitelve! Használj fit=True először.")
        
        log.info("Feature encoding (transform) folyamatban...")
        X_enc = _feature_encoder.transform(df_without_datetime)
        feature_names = _feature_encoder.get_feature_names_out()
        encoded_df = pd.DataFrame(X_enc, columns=feature_names, index=df.index)
        return encoded_df

def save_encoder(path: str = "feature_encoder.joblib"):
    """Menti a feature encodert fájlba."""
    if _feature_encoder is not None:
        joblib.dump({
            'encoder': _feature_encoder,
            'cat_cols': _cat_cols,
            'num_cols': _num_cols
        }, path)
        log.info(f"Encoder mentve: {path}")

def load_encoder(path: str = "feature_encoder.joblib"):
    """Betölti a feature encodert fájlból."""
    global _feature_encoder, _cat_cols, _num_cols
    data = joblib.load(path)
    _feature_encoder = data['encoder']
    _cat_cols = data['cat_cols']
    _num_cols = data['num_cols']
    log.info(f"Encoder betöltve: {path}")