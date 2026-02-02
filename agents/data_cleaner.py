# agents/data_cleaner.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np

log = get_logger(__name__)

class DataCleaner:
    """
    - Duplikátumok, hiányzó értékek kezelése
    - Oszlopok átnevezése (Src IP → src_ip, …)
    - Label‑kiegyensúlyozás SMOTE‑val **szűkített mintán**
    """
    def __init__(self, max_rows: int | None = None, random_state: int = 42):
        """
        Parameters
        ----------
        max_rows : int | None
            Ha megadod, a *nyers* adatból maximum ennyi sort tartunk meg
            (az elsődleges memória‑kímélő lépés).  `None` → nincs limit.
        random_state : int
            Reprodukálhatóság.
        """
        self.max_rows = max_rows
        self.random_state = random_state

    # -----------------------------------------------------------------
    # 1️⃣  Oszlop‑átnevezés (in‑place – nincs másolat)
    # -----------------------------------------------------------------
    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Src IP": "src_ip",
            "Dst IP": "dst_ip",
            "Src Port": "src_port",
            "Dst Port": "dst_port",
            "Protocol": "protocol",
            "Timestamp": "timestamp",
            "Label": "label",
        }
        df.rename(columns=rename_map, inplace=True)
        return df

    # -----------------------------------------------------------------
    # 2️⃣  Felesleges oszlopok eldobása
    # -----------------------------------------------------------------
    @staticmethod
    def _drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
        # pl. „Flow ID”, „raw_*” oszlopok (hatalmas stringek) gyakran feleslegesek
        irrelevant = [c for c in df.columns if c.startswith("Flow ID") or c.startswith("raw_")]
        return df.drop(columns=irrelevant, errors="ignore")

    # -----------------------------------------------------------------
    # 3️⃣  Timestamp konvertálása
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        # formátum: 01/01/1970 07:41:46 AM
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        return df

    # -----------------------------------------------------------------
    # 4️⃣  Hiányzó értékek kezelése (numerikus → medián, szöveg → "unknown")
    # -----------------------------------------------------------------
    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns
        
        with tqdm(total=len(numeric_cols) + len(non_numeric_cols), 
                 desc="Hiányzó értékek kezelése", 
                 bar_format='{l_bar}{bar:30}{r_bar}',
                 colour='cyan') as pbar:
            
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
                pbar.update(1)
                
            for col in non_numeric_cols:
                df[col] = df[col].fillna("unknown")
                pbar.update(1)
                
        return df

    # -----------------------------------------------------------------
    # 5️⃣  Duplikátumok eltávolítása
    # -----------------------------------------------------------------
    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        log.info(f"Duplikátumok eltávolítva: {before - len(df)}")
        return df

    # -----------------------------------------------------------------
    # 6️⃣  **Minta‑korlátozás (max_rows) – a legdrágább lépések előtt**
    # -----------------------------------------------------------------
    def _apply_max_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ha megadtuk a max_rows‑t, lecsökkentjük a DataFrame‑et.
        Ez előtt még nincs one‑hot, így a memóriafelhasználás minimális."""
        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.sample(
                n=self.max_rows,
                random_state=self.random_state,
                replace=False,
            )
            log.info(f"max_rows miatt válogatott mintára szűkítve → {len(df)} sor")
        return df

    # -----------------------------------------------------------------
    # 7️⃣  Kiegyensúlyozott osztályok (SMOTE) – csak a *kiválogatott* mintára
    # A SOTE egy olyan adatfeldolgozási eljár a gépi tanulásban, amelyet a kiegyensúlyozatlan adathalmazok kezelésére használnak. Gyakori probléma, hogy az egyik osztályból,
    # pl. támadások kevesebb mint aáll rendelkezésre mint a másikból. A SMOTE segít felduzzasztani a kissebségi osztályt szintetikus adatok generálásával.
    # -----------------------------------------------------------------
    def _balance_classes(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        # MENTÉS: dátum és időbélyeg oszlopok kivétele SMOTE előtt
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        other_non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
        other_non_numeric = [col for col in other_non_numeric if col != label_col and col not in datetime_cols]
        
        # Kivesszük ezeket az oszlopokat az SMOTE-hez
        cols_to_remove = datetime_cols + other_non_numeric
        X_for_smote = df.drop(columns=[label_col] + cols_to_remove)
        y = df[label_col]

        # Ha csak egyetlen címke van, a SMOTE értelmetlen → visszaadjuk a DF‑et
        if y.nunique() < 2:
            log.warning("Csak egyetlen címke (valószínűleg csak Benign). SMOTE kihagyva.")
            return df

        # **KATEGÓRIÁK** csak egy‑hot‑ra kerülnek,
        # numerikus oszlopok változatlanul maradnak.
        cat_cols = X_for_smote.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = [c for c in X_for_smote.columns if c not in cat_cols]

        log.info(f"Kategóriás oszlopok: {cat_cols}")
        log.info(f"Numerikus oszlopok: {len(num_cols)} (ütközés nélkül)")

        # HA NINCSENEK KATEGÓRIKUS OSZLOPOK, csak a numerikusakat használjuk
        if len(cat_cols) == 0:
            X_enc = X_for_smote[num_cols].copy()
        else:
            # One‑Hot csak a kategóriás mezőkre (memória‑kímélő)
            log.info("One‑Hot encoding folyamatban...")
            X_enc = pd.get_dummies(X_for_smote[cat_cols], drop_first=True)
            # Numerikus oszlopokat hozzáadjuk a dummy‑mátrixhoz (horizontálisan)
            if num_cols:
                X_enc = pd.concat([X_enc, X_for_smote[num_cols].reset_index(drop=True)], axis=1)

        # SMOTE az **encodált** adatként
        log.info("SMOTE kiegyensúlyozás folyamatban...")
        sm = SMOTE(random_state=self.random_state)
        X_bal, y_bal = sm.fit_resample(X_enc, y)
        log.info("SMOTE kiegyensúlyozás kész.")

        # Visszaépítjük a DataFrame‑et: SMOTE‑ul válogatott
        df_bal = pd.concat([X_bal, y_bal], axis=1)

        # VISSZARAKJUK A DATETIME ÉS EGYÉB NON-NUMERIC OSZLOOPOKAT
        if cols_to_remove:
            # Duplikáljuk a dátum oszlopokat minden új mintához
            original_datetime_data = df[cols_to_remove].iloc[:len(df_bal)]
            # Ha több sor van mint eredetileg, ismételjük meg az értékeket
            if len(df_bal) > len(original_datetime_data):
                repeat_count = (len(df_bal) // len(original_datetime_data)) + 1
                repeated_data = pd.concat([original_datetime_data] * repeat_count, ignore_index=True)
                original_datetime_data = repeated_data.iloc[:len(df_bal)]
            
            df_bal = pd.concat([df_bal.reset_index(drop=True), 
                               original_datetime_data.reset_index(drop=True)], axis=1)

        # **Végső limit** – ha a user még így is kisebb mintát akar (pl. teszt)
        if self.max_rows is not None and len(df_bal) > self.max_rows:
            df_bal, _ = train_test_split(
                df_bal,
                train_size=self.max_rows,
                stratify=df_bal[label_col],
                random_state=self.random_state,
            )
        
        log.info(f"Kiegyensúlyozott osztályok:\n{df_bal[label_col].value_counts()}")
        return df_bal

    # -----------------------------------------------------------------
    # 8️⃣  fő `run` metódus – sorrendben hívja a kisebb‑memória függvényeket
    # -----------------------------------------------------------------
    def run(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        log.info(f"Adattisztítás – eredeti sorok: {len(raw_df)}")
        
        steps = [
            "Oszlop átnevezés",
            "Felesleges oszlopok eldobása",
            "Timestamp konvertálás",
            "Hiányzó értékek kezelése",
            "Duplikátumok eltávolítása",
            "Mintakorlátozás",
            "Osztálykiegyensúlyozás"
        ]
        
        with tqdm(total=len(steps), desc="Adattisztítás", bar_format='{l_bar}{bar:30}{r_bar}', colour='magenta') as pbar:
            pbar.set_description(steps[0])
            df = self._rename_columns(raw_df)
            pbar.update(1)
            
            pbar.set_description(steps[1])
            df = self._drop_irrelevant(df)
            pbar.update(1)
            
            pbar.set_description(steps[2])
            df = self._parse_timestamp(df)
            pbar.update(1)
            
            pbar.set_description(steps[3])
            df = self._handle_missing(df)
            pbar.update(1)
            
            pbar.set_description(steps[4])
            df = self._remove_duplicates(df)
            pbar.update(1)
            
            pbar.set_description(steps[5])
            df = self._apply_max_rows(df)
            pbar.update(1)
            
            pbar.set_description(steps[6])
            df = self._balance_classes(df, label_col="label")
            pbar.update(1)
        
        log.info(f"Tisztított adathalmaz sorok: {len(df)}")
        return df