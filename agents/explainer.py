# agents/explainer.py (TELJES fájl)
import pandas as pd
import numpy as np
from utils.logger import get_logger
import ollama
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
from utils.ml_helpers import encode_features
from tqdm import tqdm
import warnings

# Próbáljuk importálni a SHAP-ot, de ha nem sikerül, kezeljük
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    log = get_logger(__name__)
    log.warning(f"SHAP nem elérhető: {e}. Alternatív magyarázatokat használunk.")
    SHAP_AVAILABLE = False

log = get_logger(__name__)

class Explainer:
    """
    - mode='shap' → SHAP‑feature‑magyarazat (RandomForest esetén)
    - mode='llm'  → LLM‑prompt‑alapú szöveges magyarázat
    - mode='simple' → Egyszerű feature importance vagy szabályalapú magyarázat
    """
    def __init__(self,
                 mode: str = "simple",  # Alapértelmezett: simple
                 llm_name: str = "llama3.1:8b-instruct",
                 llm_temp: float = 0.0):
        self.mode = mode.lower()
        self.llm_name = llm_name
        self.llm_temp = llm_temp
        self.shap_explainer = None
        
        # Ha SHAP mód, de a SHAP nem elérhető, automatikusan váltunk simple módra
        if self.mode == "shap" and not SHAP_AVAILABLE:
            log.warning("SHAP nem elérhető. Automatikusan váltás simple módra.")
            self.mode = "simple"

    # --------------------------- SHAP vagy Alternatív --------------------------- #
    def init_shap(self, model):
        if not SHAP_AVAILABLE:
            log.error("SHAP nem elérhető a rendszeren!")
            return False
            
        try:
            log.info("SHAP explainer inicializálása...")
            self.shap_explainer = shap.TreeExplainer(model)
            log.info("SHAP‑explainer inicializálva.")
            return True
        except Exception as e:
            log.error(f"SHAP inicializálási hiba: {e}")
            self.shap_explainer = None
            return False

    def explain_shap_or_simple(self, X: pd.DataFrame, model, top_n: int = 5) -> pd.Series:
        """
        SHAP magyarázat vagy alternatív feature importance alapján
        """
        # Próbáljuk meg a SHAP-ot
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                return self._explain_with_shap(X, top_n)
            except Exception as e:
                log.warning(f"SHAP számítás hiba: {e}. Alternatív módszerre váltás.")
        
        # SHAP sikertelen, használjunk alternatív módszert
        return self._explain_with_feature_importance(X, model, top_n)
    
    def _explain_with_shap(self, X: pd.DataFrame, top_n: int = 5) -> pd.Series:
        """SHAP-alapú magyarázat"""
        log.info("Feature encoding SHAP-hoz...")
        X_encoded = encode_features(X, fit=False)
        
        # OPTIMIZATION: Use a subset for large datasets
        max_shap_samples = 1000
        
        if len(X_encoded) > max_shap_samples:
            log.info(f"Túl sok sor ({len(X_encoded)}), csak {max_shap_samples} mintára számítom a SHAP értékeket")
            sample_indices = np.random.choice(len(X_encoded), size=max_shap_samples, replace=False)
            X_sample = X_encoded.iloc[sample_indices]
        else:
            X_sample = X_encoded
        
        log.info("SHAP értékek kiszámítása...")
        
        shap_vals = self.shap_explainer.shap_values(X_sample, check_additivity=False)
        
        if isinstance(shap_vals, list):   # több osztály
            shap_vals = shap_vals[1]      # index 1 = "attack"
        
        explanations = []
        feature_names = X_sample.columns.tolist()
        
        log.info("SHAP magyarázatok generálása...")
        for i, row_vals in enumerate(tqdm(shap_vals, total=len(shap_vals), desc="SHAP magyarázatok", 
                                        unit="sor", bar_format='{l_bar}{bar:30}{r_bar}', colour='cyan')):
            idx = np.argsort(np.abs(row_vals))[-top_n:][::-1]
            feats = []
            for feat_idx in idx:
                feat_name = feature_names[feat_idx]
                # Egyszerűsítjük a feature neveket
                if feat_name.startswith("src_ip_"):
                    feats.append(f"forrás IP ({row_vals[feat_idx]:+.3f})")
                elif feat_name.startswith("dst_ip_"):
                    feats.append(f"cél IP ({row_vals[feat_idx]:+.3f})")
                elif feat_name.startswith("protocol_"):
                    feats.append(f"protokoll ({row_vals[feat_idx]:+.3f})")
                else:
                    # Numerikus feature-ök
                    feats.append(f"{feat_name} ({row_vals[feat_idx]:+.3f})")
            
            explanations.append(", ".join(feats))
        
        # Ha mintavételezettünk, terjesszük ki az összes sorra
        if len(X_encoded) > max_shap_samples:
            full_explanations = ["SHAP alapú elemzés"] * len(X)
            for i, sample_idx in enumerate(sample_indices):
                full_explanations[sample_idx] = explanations[i]
            return pd.Series(full_explanations, index=X.index)
        else:
            return pd.Series(explanations, index=X.index)
    
    def _explain_with_feature_importance(self, X: pd.DataFrame, model, top_n: int = 5) -> pd.Series:
        """Feature importance alapú magyarázat (SHAP alternatíva)"""
        log.info("Feature importance alapú magyarázat generálása...")
        
        # Szerezzük meg a feature importance-t
        if hasattr(model, 'feature_importances_'):
            # Encode features to get the right feature names
            X_encoded = encode_features(X, fit=False)
            
            # Get feature importances
            importances = model.feature_importances_
            feature_names = X_encoded.columns
            
            # Rendezzük a feature-öket importance szerint
            sorted_idx = np.argsort(importances)[::-1]
            top_features = [(feature_names[i], importances[i]) for i in sorted_idx[:top_n]]
            
            # Generáljunk általános magyarázatot a legfontosabb feature-ökre
            base_explanation = "Legfontosabb jellemzők: "
            for feat_name, importance in top_features:
                simple_name = self._simplify_feature_name(feat_name)
                base_explanation += f"{simple_name}({importance:.3f}), "
            
            base_explanation = base_explanation.rstrip(", ")
            
            # Minden sorra ugyanazt a magyarázatot adjuk
            explanations = [base_explanation] * len(X)
            return pd.Series(explanations, index=X.index)
        else:
            # Ha nincs feature importance, akkor egyszerű szabály alapú magyarázat
            return self._explain_with_rules(X)
    
    def _explain_with_rules(self, X: pd.DataFrame) -> pd.Series:
        """Egyszerű szabály alapú magyarázat"""
        log.info("Szabályalapú magyarázat generálása...")
        
        explanations = []
        
        for idx, row in X.iterrows():
            features = []
            
            # Néhány egyszerű szabály
            if 'n_events' in row:
                n_events = row['n_events']
                if n_events > 100:
                    features.append(f"magas eseményszám ({int(n_events)})")
                elif n_events < 3:
                    features.append(f"alacsony eseményszám ({int(n_events)})")
            
            if 'duration_sec' in row:
                duration = row['duration_sec']
                if duration > 300:  # 5+ minutes
                    features.append(f"hosszú időtartam ({duration:.0f}s)")
                elif duration < 1:
                    features.append(f"rövid időtartam ({duration:.3f}s)")
            
            if 'dst_port' in row:
                dst_port = row['dst_port']
                if dst_port in [22, 23, 3389, 5900, 21, 25, 53, 80, 443]:
                    features.append(f"ismert port ({int(dst_port)})")
                elif dst_port < 1024:
                    features.append(f"rendszer port ({int(dst_port)})")
            
            if 'src_port' in row:
                src_port = row['src_port']
                if src_port > 49151:
                    features.append(f"magas forrásport ({int(src_port)})")
            
            # Protocol check
            if 'protocol' in row:
                protocol = row['protocol']
                if protocol in [6, 17]:  # TCP vagy UDP
                    features.append(f"protokoll {int(protocol)}")
            
            # Risk score check (ha van)
            if 'risk_score' in row:
                risk = row['risk_score']
                if risk > 5:
                    features.append(f"magas kockázat ({risk:.1f})")
            
            if features:
                explanations.append("; ".join(features))
            else:
                explanations.append("Normális hálózati forgalom")
        
        return pd.Series(explanations, index=X.index)
    
    @staticmethod
    def _simplify_feature_name(feat_name: str) -> str:
        """Feature nevek egyszerűsítése"""
        if feat_name.startswith("src_ip_"):
            return "forrás IP"
        elif feat_name.startswith("dst_ip_"):
            return "cél IP"
        elif feat_name.startswith("protocol_"):
            return "protokoll"
        elif "port" in feat_name.lower():
            return "port"
        elif "duration" in feat_name.lower():
            return "időtartam"
        elif "event" in feat_name.lower():
            return "eseményszám"
        elif "risk" in feat_name.lower():
            return "kockázat"
        else:
            return feat_name

    # --------------------------- LLM --------------------------- #
    @staticmethod
    def _prompt(row: pd.Series, pred: str, conf: float) -> str:
        # Build feature list excluding non-feature columns
        exclude_cols = ["prediction", "confidence", "explanation", "explanation_llm", 
                       "session_id", "label_majority", "start_time", "end_time"]
        feature_cols = [c for c in row.index if c not in exclude_cols]
        
        # Select only top 5 features for brevity
        if len(feature_cols) > 5:
            # Try to get the most important features
            numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(row[c])]
            if numeric_cols:
                # Sort by absolute value (most important)
                sorted_cols = sorted(numeric_cols, key=lambda x: abs(row[x]) if not pd.isna(row[x]) else 0, reverse=True)
                feature_cols = sorted_cols[:5]
            else:
                feature_cols = feature_cols[:5]
        
        feats = "\n".join([f"- {c}: {row[c]}" for c in feature_cols])
        tmpl = (
            "Az alábbi IDS session részletei:\n{features}\n"
            "A modell előrejelzése: {pred} (bizalom {conf:.2%}).\n"
            "Adj egy 1‑2 mondatos, laikusoknak is érthető magyarázatot."
        )
        return tmpl.format(features=feats, pred=pred, conf=conf)

    def explain_llm(self, df: pd.DataFrame) -> pd.Series:
        exps = []
        log.info(f"LLM magyarázatok generálása {len(df)} sorhoz...")
        
        # Progress bar az LLM hívásokhoz
        pbar = tqdm(total=len(df), desc="LLM magyarázatok", unit="sor", 
                   bar_format='{l_bar}{bar:30}{r_bar}', colour='magenta')
        
        for _, row in df.iterrows():
            prompt = self._prompt(row, row["prediction"], row["confidence"])
            try:
                resp = ollama.generate(
                    model=self.llm_name,
                    prompt=prompt,
                    options={'temperature': self.llm_temp},
                    stream=False,
                )["response"]
                # Röviddé vágjuk és megtisztítjuk
                clean_resp = resp.strip()[:150]
                if len(clean_resp) >= 150:
                    clean_resp += "..."
                exps.append(clean_resp)
            except Exception as e:
                log.warning(f"LLM hiba: {e}")
                exps.append("LLM magyarázat nem elérhető")
            pbar.update(1)
            pbar.set_postfix({"elemzett": pbar.n})
        
        pbar.close()
        return pd.Series(exps, index=df.index)

    # --------------------------- KÖZÖS --------------------------- #
    def run(self,
            df: pd.DataFrame,
            model: RandomForestClassifier | None = None) -> pd.DataFrame:
        """
        `df` már tartalmazza a `prediction` és `confidence` oszlopokat.
        """
        log.info(f"Magyarázatok generálása {self.mode} módban...")
        
        if self.mode == "shap":
            if model is None:
                raise ValueError("SHAP‑mód: a model argument hiányzik.")
            
            # Inicializáljuk a SHAP-ot
            shap_success = self.init_shap(model)
            
            if shap_success:
                # Prepare X by removing prediction and confidence columns
                exclude_cols = ["prediction", "confidence"]
                X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
                
                # Get SHAP or alternative explanations
                df["explanation"] = self.explain_shap_or_simple(X, model)
            else:
                # SHAP sikertelen, használjunk alternatív módszert
                log.warning("SHAP sikertelen, alternatív magyarázatra váltás.")
                exclude_cols = ["prediction", "confidence"]
                X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
                df["explanation"] = self._explain_with_feature_importance(X, model)
                
        elif self.mode == "llm":
            df["explanation"] = self.explain_llm(df)
            
        elif self.mode == "simple":
            # Egyszerű szabályalapú magyarázat
            exclude_cols = ["prediction", "confidence"]
            X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
            
            if model is not None and hasattr(model, 'feature_importances_'):
                df["explanation"] = self._explain_with_feature_importance(X, model)
            else:
                df["explanation"] = self._explain_with_rules(X)
        
        else:
            raise ValueError(f"Ismeretlen mód: {self.mode}")
        
        log.info(f"Magyarázatok elkészítve ({len(df)} sor).")
        return df