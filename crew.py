# crew.py
from agents.data_cleaner import DataCleaner
from agents.correlator import Correlator
from agents.detector import Detector
from agents.explainer import Explainer
from agents.investigator import Investigator
from utils.logger import get_logger
from tqdm import tqdm
import time
import pandas as pd  # <-- EZ HIÁNYZOTT!

log = get_logger(__name__)

class Crew:
    """
    Az összes ügynök koordinálása.
    """
    def __init__(self,
                 cleaner_kwargs=None,
                 correlator_kwargs=None,
                 detector_kwargs=None,
                 explainer_kwargs=None,
                 investigator_kwargs=None):
        # Alapértelmezett értékek
        cleaner_kwargs = cleaner_kwargs or {}
        correlator_kwargs = correlator_kwargs or {}
        detector_kwargs = detector_kwargs or {}
        explainer_kwargs = explainer_kwargs or {}
        investigator_kwargs = investigator_kwargs or {}
        
        # Ha explainer mód nem megadott, alapértelmezett "simple"
        if 'mode' not in explainer_kwargs:
            explainer_kwargs['mode'] = 'simple'
            
        self.cleaner = DataCleaner(**cleaner_kwargs)
        self.correlator = Correlator(**correlator_kwargs)
        self.detector = Detector(**detector_kwargs)
        self.explainer = Explainer(**explainer_kwargs)
        self.investigator = Investigator(**investigator_kwargs)
        
        # Teljesítmény mérés
        self.execution_times = {}

    def _run_with_timing(self, func, *args, **kwargs):
        """Függvény futtatása időméréssel"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.execution_times[func_name] = end_time - start_time
        
        return result

    def run(self, raw_df, train_detector=False):
        log.info("=== Crew indítása ===")
        log.info(f"Bemeneti adatok: {len(raw_df)} sor")
        
        # Teljes folyamat progress bar
        steps = [
            "1/5 - Adattisztítás",
            "2/5 - Session építés", 
            "3/5 - Detektálás",
            "4/5 - Magyarázat",
            "5/5 - Incidenskártya"
        ]
        
        # Color mapping for different steps
        colors = ['green', 'blue', 'yellow', 'magenta', 'cyan']
        
        overall_progress = tqdm(
            total=len(steps), 
            desc="Teljes folyamat",
            bar_format='{l_bar}{bar:40}{r_bar}',
            colour='white',
            position=0,
            leave=True
        )
        
        results = {}
        
        try:
            # 1️⃣ Adattisztítás
            overall_progress.set_description(steps[0])
            overall_progress.colour = colors[0]
            log.info("=== Adattisztítás kezdete ===")
            clean = self._run_with_timing(self.cleaner.run, raw_df)
            results["clean"] = clean
            overall_progress.update(1)
            log.info(f"Adattisztítás kész: {len(clean)} sor")
            
            # 2️⃣ Session‑építés
            overall_progress.set_description(steps[1])
            overall_progress.colour = colors[1]
            log.info("=== Session építés kezdete ===")
            sessions = self._run_with_timing(self.correlator.run, clean)
            results["sessions"] = sessions
            overall_progress.update(1)
            log.info(f"Session építés kész: {len(sessions)} session")
            
            # 3️⃣ Detektálás (ml vagy llm)
            overall_progress.set_description(steps[2])
            overall_progress.colour = colors[2]
            log.info("=== Detektálás kezdete ===")
            log.info(f"Detektor mód: {self.detector.mode}")
            
            detections = self._run_with_timing(
                self.detector.run,
                sessions,
                train=train_detector,
                label_col="label_majority"
            )
            
            results["detections"] = detections
            overall_progress.update(1)
            
            # Ellenőrizzük, hogy van-e prediction oszlop
            if 'prediction' not in detections.columns:
                log.error("A detektálás nem hozta létre a 'prediction' oszlopot!")
                # Hozzunk létre egy placeholder oszlopot
                detections['prediction'] = 'unknown'
                detections['confidence'] = 0.0
                
            log.info(f"Detektálás kész: {len(detections)} detekció")
            log.info(f"Előrejelzések eloszlása:\n{detections['prediction'].value_counts()}")
            
            # 4️⃣ Magyarázat (SHAP vagy LLM vagy simple)
            overall_progress.set_description(steps[3])
            overall_progress.colour = colors[3]
            log.info("=== Magyarázat kezdete ===")
            log.info(f"Magyarázó mód: {self.explainer.mode}")
            
            # Ellenőrizzük, hogy van-e modell ML módban
            model_to_pass = None
            if self.detector.mode == "ml" and hasattr(self.detector, 'model'):
                model_to_pass = self.detector.model
                log.info(f"Modell átadva a magyarázónak: {type(model_to_pass).__name__}")
            
            detections_with_explanation = self._run_with_timing(
                self.explainer.run,
                detections,
                model=model_to_pass
            )
            
            results["detections"] = detections_with_explanation
            overall_progress.update(1)
            
            # Ellenőrizzük az explanation oszlopot
            if 'explanation' not in detections_with_explanation.columns:
                log.warning("A magyarázó nem hozta létre az 'explanation' oszlopot!")
                detections_with_explanation['explanation'] = 'Nincs magyarázat'
            else:
                # Statisztika a magyarázatokról
                explanation_counts = detections_with_explanation['explanation'].value_counts()
                log.info(f"Magyarázatok generálva: {len(explanation_counts)} egyedi magyarázat")
                if len(explanation_counts) > 0:
                    log.info(f"Leggyakoribb magyarázat: {explanation_counts.index[0][:50]}...")
            
            # 5️⃣ Incidenskártya
            overall_progress.set_description(steps[4])
            overall_progress.colour = colors[4]
            log.info("=== Incidenskártya kezdete ===")
            
            # Ellenőrizzük a confidence oszlopot
            if 'confidence' not in detections_with_explanation.columns:
                log.warning("Nincs 'confidence' oszlop, alapértelmezett értékek használata")
                detections_with_explanation['confidence'] = 0.5
            
            report = self._run_with_timing(self.investigator.run, detections_with_explanation)
            results["report"] = report
            overall_progress.update(1)
            
            overall_progress.close()
            
            # Teljesítmény összegzés
            log.info("=== Végrehajtási idők ===")
            total_time = sum(self.execution_times.values())
            for step_name, step_time in self.execution_times.items():
                percentage = (step_time / total_time) * 100
                log.info(f"  {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
            log.info(f"  Összesen: {total_time:.2f}s")
            
            # Eredmények összegzése
            log.info("=== Eredmények összegzése ===")
            log.info(f"  Tisztított adatok: {len(results['clean'])} sor")
            log.info(f"  Session-ek: {len(results['sessions'])}")
            log.info(f"  Detekciók: {len(results['detections'])}")
            
            if 'prediction' in results['detections'].columns:
                pred_counts = results['detections']['prediction'].value_counts()
                for pred, count in pred_counts.items():
                    log.info(f"    {pred}: {count} ({count/len(results['detections'])*100:.1f}%)")
            
            log.info(f"  Incidenskártyák: {len(results['report'])}")
            
            # Ha van report, mutassuk az első néhány incidens típusát
            if len(results['report']) > 0 and 'type' in results['report'].columns:
                incident_types = results['report']['type'].value_counts()
                for inc_type, count in incident_types.items():
                    log.info(f"    {inc_type}: {count} incidens")
            
            log.info("=== Crew befejeződött ===")
            
            return {
                "clean": results["clean"],
                "sessions": results["sessions"],
                "detections": results["detections"],
                "report": results["report"],
                "execution_times": self.execution_times,
                "statistics": {
                    "clean_rows": len(results["clean"]),
                    "sessions_count": len(results["sessions"]),
                    "detections_count": len(results["detections"]),
                    "incidents_count": len(results["report"]),
                    "predictions": dict(results["detections"]['prediction'].value_counts()) if 'prediction' in results["detections"].columns else {},
                    "incident_types": dict(results["report"]['type'].value_counts()) if len(results["report"]) > 0 and 'type' in results["report"].columns else {}
                }
            }
            
        except Exception as e:
            overall_progress.close()
            log.error(f"Hiba a Crew futása közben: {e}", exc_info=True)
            
            # Visszaadjuk az eddigi eredményeket
            if 'clean' in results:
                log.info("Visszaadom a részleges eredményeket...")
                return {
                    "clean": results.get("clean", pd.DataFrame()),
                    "sessions": results.get("sessions", pd.DataFrame()),
                    "detections": results.get("detections", pd.DataFrame()),
                    "report": results.get("report", pd.DataFrame()),
                    "error": str(e),
                    "execution_times": self.execution_times
                }
            else:
                # Ha még nincs eredmény, dobjuk tovább a hibát
                raise