# ==============================================================
#  main_windows.py – Windows‑barát indító a Crew‑AI‑pipeline‑hoz
# ==============================================================

import argparse
import pandas as pd
from pathlib import Path
from crew import Crew
from utils.logger import init_logging, get_logger
from tqdm import tqdm

log = get_logger(__name__)

# --------------------------------------------------------------
# 1️⃣  CHUNK‑OLVASÓ – memóriakímélő CSV‑betöltés
# --------------------------------------------------------------
def load_csv_chunked(
    path: str,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    chunksize: int = 250_000,
    engine: str = "python",
    on_bad_lines: str = "skip",
    max_rows: int | None = None         # ← új paraméter
) -> pd.DataFrame:
    """
    Chunk‑olvasás, a `max_rows`‑ként megadott sorlimit alkalmazva.
    - Ha `max_rows` = None → a teljes fájlt beolvassa (a régi viselkedés).
    - Ha megadod, a ciklus **megáll** a limitnél, és a felesleges
      sorokat a legutolsó chunk‑ból vágja le.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"CSV fájl nem található: {path}")

    log.info(
        f"CSV betöltése chunk‑onként "
        f"({chunksize} sor/chunk, sep='{sep}', engine='{engine}')"
    )
    chunks = []
    rows_sofar = 0

    # Becslés a progress bar-hoz
    try:
        if max_rows is not None:
            estimated_chunks = (max_rows // chunksize) + 1
        else:
            # Fájlméret alapján becsüljük
            file_size = path_obj.stat().st_size
            estimated_chunks = (file_size // (chunksize * 200)) + 1 if file_size > 0 else None
    except:
        estimated_chunks = None

    chunk_iterator = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        chunksize=chunksize,
        engine=engine,
        on_bad_lines=on_bad_lines,
    )

    # Progress bar a chunk betöltéshez
    pbar = tqdm(
        desc="CSV betöltés",
        unit="chunk", 
        total=estimated_chunks,
        bar_format='{l_bar}{bar:30}{r_bar}',
        colour='blue'
    )

    for i, chunk in enumerate(chunk_iterator):
        # Ha már elértük (vagy meghaladtuk) a limitet, csak a szükséges részt vegyük
        if max_rows is not None:
            remaining = max_rows - rows_sofar
            if remaining <= 0:
                # már nincs több sorra szükség → kilépünk a ciklusból
                log.debug(f"max_rows ({max_rows}) elérve – a többi chunk kihagyva.")
                break
            if len(chunk) > remaining:
                # csak annyi sort vesszük a chunk‑ból, amennyi még hiányzik
                chunk = chunk.iloc[:remaining]
                log.debug(
                    f"Chunk {i+1} – csak {remaining} sorra vágva (max_rows={max_rows})"
                )
                chunks.append(chunk)
                rows_sofar += len(chunk)
                pbar.update(1)
                pbar.set_postfix({"sorok": rows_sofar})
                # limit teljes – kilépünk
                break
        
        # limit nincs vagy még nem telt el → a teljes chunk‑ot hozzáadjuk
        chunks.append(chunk)
        rows_sofar += len(chunk)
        pbar.update(1)
        pbar.set_postfix({"sorok": rows_sofar})
        
        log.debug(f"Chunk {i+1} beolvasva – sorok: {len(chunk)} (összes: {rows_sofar})")

    pbar.close()
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"CSV betöltve – összes sor (limit {max_rows}): {len(df)}")
    return df

# --------------------------------------------------------------
# 2️⃣  CLI – argumentum‑parser + fő függvény
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IDS‑naplófeldolgozó – Crew‑AI (Windows) – "
                    "ml = RandomForest, llm = Ollama‑prompt"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Bemeneti CSV (pl. D:\\ids_crew\\data\\data.csv)"
    )
    parser.add_argument(
        "-m", "--mode", choices=["ml", "llm"], default="ml",
        help="Detektor mód: 'ml' (RandomForest) vagy 'llm' (LLM‑prompt)."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="RandomForest modell betanítása (csak –m ml esetén)."
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Adattisztító limit – ha a CSV túl nagy, korlátozhatók a sorok."
    )
    args = parser.parse_args()

    # ----------------------------------------------------------
    # 3️⃣  LOGOLÁS inicializálása (DEBUG‑szint)
    # ----------------------------------------------------------
    init_logging(level=10)          # DEBUG = 10
    log.info("=== Crew‑AI pipeline indítása ===")
    log.info(f"Bemeneti fájl: {args.input}")

    # ----------------------------------------------------------
    # 4️⃣  CSV betöltése (chunk‑olvasó) – itt nem használunk usecols‑t
    # ----------------------------------------------------------
    raw_df = load_csv_chunked(
        args.input,
        sep=",",
        encoding="utf-8",
        chunksize=250_000,
        engine="python",
        on_bad_lines="skip",
        max_rows=args.max_rows          # ← itt adod át a parancssori flag‑et
    )

    # ----------------------------------------------------------
    # 5️⃣  Crew objektum összeállítása
    # ----------------------------------------------------------
    crew = Crew(
        cleaner_kwargs={"max_rows": args.max_rows},
        correlator_kwargs={"time_window": pd.Timedelta(minutes=5), "min_events": 1},
        detector_kwargs={"mode": args.mode},
        explainer_kwargs={"mode": "simple"},
        investigator_kwargs={"min_confidence": 0.1, "max_reports": 200},
    )

    # ----------------------------------------------------------
    # 6️⃣  Pipeline futtatása
    # ----------------------------------------------------------
    result = crew.run(raw_df, train_detector=args.train)

    # ----------------------------------------------------------
    # 7️⃣  Kimeneti fájlok mentése a projekt gyökérkönyvtárába
    # ----------------------------------------------------------
    with tqdm(total=4, desc="Eredmények mentése", bar_format='{l_bar}{bar:30}{r_bar}', colour='yellow') as pbar:
        result["clean"].to_csv("out_clean.csv", index=False)
        pbar.update(1)
        pbar.set_description("Eredmények mentése (1/4)")
        
        result["sessions"].to_csv("out_sessions.csv", index=False)
        pbar.update(1)
        pbar.set_description("Eredmények mentése (2/4)")
        
        result["detections"].to_csv("out_detections.csv", index=False)
        pbar.update(1)
        pbar.set_description("Eredmények mentése (3/4)")
        
        result["report"].to_csv("incidents_report.csv", index=False)
        pbar.update(1)
        pbar.set_description("Eredmények mentése (4/4)")

    log.info("=== Pipeline befejeződött – CSV‑k mentve ===")
    log.info("  out_clean.csv")
    log.info("  out_sessions.csv")
    log.info("  out_detections.csv")
    log.info("  incidents_report.csv")
    log.info("Az eredmény megtekinthető:  Invoke-Item .\\incidents_report.csv")

if __name__ == "__main__":
    main()