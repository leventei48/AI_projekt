import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.ml_helpers import encode_features
from utils.logger import get_logger
import ollama
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

log = get_logger(__name__)
 #random forest, mert több döntési fát egyesít egy modellbe, így javítva a pontosságot és csökkentve a túltanulás esélyét.
class Detector:
    """
    - mode='ml' : scikit‑learn RandomForest (kérdés alapján)
    - mode='llm': Ollama‑prompt (újra‑detectálás)
    """
    def __init__(self,
                 mode: str = "ml",
                 model: RandomForestClassifier | None = None, 
                 llm_name: str = "llama3.1",
                 llm_temp: float = 0.2): # Alacsonyabb hőmérséklet a kiszámíthatóbb válaszokért
        self.mode = mode.lower()
        self.llm_name = llm_name
        self.llm_temp = llm_temp

        if self.mode == "ml": #Random forest adatainak beállitása
            self.model = model or RandomForestClassifier(
                n_estimators=200, 
                n_jobs=-1, 
                random_state=42, 
                class_weight="balanced", 
                verbose=0
            )
        elif self.mode == "llm":
            self.model = None
        else:
            raise ValueError("mode must be 'ml' vagy 'llm'")

    # --------------------------- ML --------------------------- #
    def train(self, X: pd.DataFrame, y: pd.Series):
        log.info("Feature encoding folyamatban...")
        X_enc = encode_features(X, fit=True)
        
        log.info(f"RandomForest betanítása {len(X_enc)} mintára...")
        
        # Egyszerű progress bar a betanításhoz
        with tqdm(total=100, desc="Modell betanítása", bar_format='{l_bar}{bar:30}{r_bar}', colour='green') as pbar: 
            self.model.fit(X_enc, y)
            pbar.update(100)
            
        log.info(f"RandomForest betanítva. Fák: {self.model.n_estimators}")

    def predict_ml(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = encode_features(X, fit=False)
        
        log.info(f"Előrejelzés {len(X_enc)} mintára...")
        
        # Progress bar az előrejelzéshez
        with tqdm(total=100, desc="Előrejelzés", bar_format='{l_bar}{bar:30}{r_bar}', colour='yellow') as pbar:
            probs = self.model.predict_proba(X_enc)
            pbar.update(30)
            
            pred_idx = np.argmax(probs, axis=1)
            pbar.update(30)
            
            preds = self.model.classes_[pred_idx]
            confidence = probs.max(axis=1)
            pbar.update(40)

        out = X.copy()
        out["prediction"] = preds
        out["confidence"] = confidence
        return out

    # --------------------------- LLM --------------------------- #
    @staticmethod
    def _row2text(row: pd.Series) -> str:
        # Csak a legfontosabb mezőket használjuk
        features = {
            'src_ip': row.get('src_ip', 'N/A'),
            'dst_ip': row.get('dst_ip', 'N/A'),
            'src_port': row.get('src_port', 'N/A'),
            'dst_port': row.get('dst_port', 'N/A'),
            'protocol': row.get('protocol', 'N/A'),
            'duration_sec': f"{row.get('duration_sec', 0):.0f}" if pd.notna(row.get('duration_sec')) else 'N/A',
            'n_events': row.get('n_events', 'N/A'),
        }
        
        tmpl = "src_ip={src_ip} dst_ip={dst_ip} src_port={src_port} dst_port={dst_port} protocol={protocol} duration={duration_sec}s n_events={n_events}"
        return tmpl.format(**features)

    def _detect_llm(self, df: pd.DataFrame) -> pd.DataFrame:
        def ask(row_txt: str) -> tuple[str, float, str]:
            prompt = (
                "Az alábbi IDS session adatai:\n"
                f"{row_txt}\n"
                "Válasz csak a következő formátumban:\n"
                "LABEL: <benign|attack>\n"
                "CONF: <0‑1>  (bizalom)\n"
                "EXPL: <rövid indoklás>\n"
                "Példa: LABEL: attack; CONF: 0.96; EXPL: Szokatlan port‑kombináció."
            )
            
            try:
                # ÚJ OLLAMA API - temperature az options dict-ben van
                resp = ollama.generate(
                    model=self.llm_name,
                    prompt=prompt,
                    options={'temperature': self.llm_temp},  # <-- IDE KERÜL
                    stream=False,
                )
                
                response_text = resp['response']
                
                # Parsoljuk a választ
                try:
                    # Tisztítsuk a választ
                    clean_response = response_text.strip()
                    
                    # Keressük a LABEL, CONF, EXPL részleteket
                    label = "unknown"
                    conf = 0.0
                    expl = "parsing error"
                    
                    # Próbáljuk kinyerni az információkat
                    if 'LABEL:' in clean_response:
                        parts = clean_response.split('\n')
                        for part in parts:
                            part = part.strip()
                            if part.startswith('LABEL:'):
                                label = part.replace('LABEL:', '').strip().lower()
                            elif part.startswith('CONF:'):
                                try:
                                    conf_str = part.replace('CONF:', '').strip()
                                    conf = float(conf_str)
                                except:
                                    conf = 0.5
                            elif part.startswith('EXPL:'):
                                expl = part.replace('EXPL:', '').strip()
                    
                    # Ha nem találtuk a formátumot, próbáljunk másik módszert
                    if label == "unknown":
                        # Egyszerű keresés
                        if 'attack' in clean_response.lower():
                            label = "attack"
                            conf = 0.7
                        elif 'benign' in clean_response.lower():
                            label = "benign"
                            conf = 0.7
                        else:
                            # Alapértelmezett
                            label = "benign"
                            conf = 0.5
                            expl = "Nem egyértelmű, alapértelmezett benign"
                            
                except Exception as parse_error:
                    log.debug(f"Parse hiba: {parse_error}")
                    label, conf, expl = "unknown", 0.5, "parsing error"
                    
                return label, conf, expl
                
            except Exception as e:
                log.warning(f"LLM hívási hiba: {e}")
                return "unknown", 0.0, f"LLM error: {str(e)[:50]}"

        results = []
        log.info(f"LLM elemzés {len(df)} session-re...")
        
        # Progress bar az LLM hívásokhoz
        pbar = tqdm(total=len(df), desc="LLM elemzés", unit="session", 
                   bar_format='{l_bar}{bar:30}{r_bar}', colour='cyan')
        
        for _, row in df.iterrows():
            txt = self._row2text(row)
            results.append(ask(txt))
            pbar.update(1)
            pbar.set_postfix({"elemzett": pbar.n})

        pbar.close()
        
        # Ellenőrizzük, hogy van-e eredmény
        if not results:
            log.warning("Nincs LLM válasz!")
            # Alapértelmezett értékek
            lbls = ["benign"] * len(df)
            confs = [0.5] * len(df)
            exps = ["LLM nem válaszolt"] * len(df)
        else:
            lbls, confs, exps = zip(*results)
            
        out = df.copy()
        out["prediction"] = lbls
        out["confidence"] = confs
        out["explanation_llm"] = exps
        
        # Statisztikák
        log.info(f"LLM elemzés kész. Előrejelzések:")
        pred_counts = out['prediction'].value_counts()
        for pred, count in pred_counts.items():
            percentage = (count / len(out)) * 100
            log.info(f"  {pred}: {count} ({percentage:.1f}%)")
        
        return out

    # --------------------------- KÖZÖS --------------------------- #
    def run(self,
            df: pd.DataFrame,
            train: bool = False,
            label_col: str = "label"):
        """
        `train=True` → csak ML‑módban használatos (betanítás).
        Visszaad egy DataFrame‑et, ahol legalább a `prediction`
        és `confidence` oszlopok benne vannak.
        """
        if self.mode == "ml":
            X = df.drop(columns=[label_col])
            y = df[label_col]
            
            if train:
                self.train(X, y)
                
            return self.predict_ml(X)
        else:  # llm
            return self._detect_llm(df)