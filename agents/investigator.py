# agents/investigator.py
import pandas as pd
from utils.logger import get_logger
from tqdm import tqdm

log = get_logger(__name__)

class Investigator:
    """
    Incidenskártyák generálása a detektált sorokból.
    """
    def __init__(self,
                 min_confidence: float = 0.65,
                 max_reports: int = 20):
        self.min_conf = min_confidence
        self.max_reports = max_reports

    @staticmethod
    def _format_ts(ts):
        if isinstance(ts, pd.Timestamp):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info(f"Incidenskártyák generálása...")
        
        # 1️⃣ szűrés bizalom alapján
        cand = df[df["confidence"] >= self.min_conf].copy()
        log.info(f"{len(cand)} sor marad a {self.min_conf:.2%} bizalmi szint után.")
        
        if len(cand) == 0:
            log.warning("Nincs elég magas bizalommal rendelkező detekció!")
            # Üres DataFrame visszaadása a megfelelő oszlopokkal
            return pd.DataFrame(columns=[
                "incident_id", "time", "src_ip", "src_port",
                "dst_ip", "dst_port", "protocol", "type",
                "confidence_score", "explanation"
            ])

        # 2️⃣ csoportosítás (src/dst/port/protocol) progress bar-ral
        group_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
        
        # Először csoportosítás
        log.info("Incidensek csoportosítása...")
        grouped = cand.groupby(group_cols)
        
        # Aggregáció progress bar-ral
        reports = []
        pbar = tqdm(total=len(grouped), desc="Incidensek aggregálása", 
                   unit="csoport", bar_format='{l_bar}{bar:30}{r_bar}', colour='yellow')
        
        for name, group in grouped:
            report_row = {
                "src_ip": name[0],
                "dst_ip": name[1],
                "src_port": name[2],
                "dst_port": name[3],
                "protocol": name[4],
                "confidence": group["confidence"].max(),
                "prediction": group["prediction"].iloc[0] if len(group) > 0 else "unknown",
                "explanation": group["explanation"].iloc[0] if len(group) > 0 else "nincs",
                "timestamp": group["start_time"].min() if "start_time" in group.columns else group.index[0],
            }
            reports.append(report_row)
            pbar.update(1)
        
        pbar.close()
        report = pd.DataFrame(reports)

        # 3️⃣ rangsorolás & limitálás
        if len(report) > 0:
            report = report.sort_values("confidence", ascending=False).head(self.max_reports)

            # 4️⃣ végső formátum
            report["incident_id"] = range(1, len(report) + 1)
            report["time"] = report["timestamp"].apply(self._format_ts)
            report = report.rename(columns={
                "prediction": "type",
                "confidence": "confidence_score"
            })
            
            final_cols = [
                "incident_id", "time", "src_ip", "src_port",
                "dst_ip", "dst_port", "protocol", "type",
                "confidence_score", "explanation"
            ]
            
            # Ellenőrizzük, hogy minden oszlop létezik
            existing_cols = [col for col in final_cols if col in report.columns]
            missing_cols = [col for col in final_cols if col not in report.columns]
            
            if missing_cols:
                log.warning(f"Hiányzó oszlopok: {missing_cols}")
                for col in missing_cols:
                    report[col] = "N/A"
            
            report = report[final_cols]
        else:
            report = pd.DataFrame(columns=final_cols)
        
        log.info(f"Végső incidenskártyák száma: {len(report)}")
        return report