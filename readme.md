# IDS Log Analysis Pipeline with CrewAI

## 📋 Projekt Áttekintés

Ez a projekt egy **több-ügynökös rendszer (Multi-Agent System)** a CrewAI keretrendszer mintájára, amely hálózati IDS (Intrusion Detection System) naplók automatikus elemzésére és fenyegetésészlelésére szolgál. A rendszer 5 specializált ügynökön keresztül dolgozza fel a naplóadatokat, hogy incidenseket észleljen és jelentéseket készítsen.

## 🎯 Főbb Funkciók

- **Adattisztítás**: Nagyméretű CSV fájlok feldolgozása (akár több millió sor)
- **Session építés**: Időbeli korreláció alapján hálózati session-ek azonosítása
- **Fenyegetésészlelés**: ML (RandomForest) vagy LLM (Ollama) alapú detektálás
- **Magyarázat**: Észlelések értelmezése emberi nyelven (SHAP, LLM vagy simple mód)
- **Incidensjelentés**: Részletes kártyák generálása a detektált fenyegetésekről

## 📁 Fájlstruktúra

```
ids_crew/
├── agents/
│   ├── data_cleaner.py      # Adattisztító ügynök
│   ├── correlator.py        # Korrelátor ügynök (időablakos session építés)
│   ├── detector.py          # Detektor ügynök (ML vagy LLM alapú)
│   ├── explainer.py         # Magyarázó ügynök
│   └── investigator.py      # Incidenskezelő ügynök
├── utils/
│   ├── ml_helpers.py        # ML segédfüggvények (feature encoding)
│   └── logger.py           # Naplózó konfiguráció
├── crew.py                 # Fő csapat koordinátor
├── main_windows.py         # Fő indító script
├── README.md              # Ez a fájl
└── data/
    └── data.csv           # Bemeneti IDS naplófájl (példa)
```

## 🚀 Gyors Start

### Előfeltételek
```bash
# Python csomagok telepítése
pip install -r requirements.txt

# Opcionális: SHAP (magyarázatokhoz)
pip install shap

# Opcionális: Ollama (LLM módhoz)
# Letöltés: https://ollama.com/download
ollama pull llama3
```

### Futtatás

#### 1. **ML módban (RandomForest) - AJÁNLOTT**
```bash
# Betanítással és előrejelzéssel
python main_windows.py -i data/data.csv -m ml --train --max-rows 50000

# Csak előrejelzés (ha már van betanított modell)
python main_windows.py -i data/data.csv -m ml --max-rows 10000
```

#### 2. **LLM módban (Ollama) - KÍSÉRLETI**
```bash
# Kis adatmennyiséggel tesztelés
python main_windows.py -i data/data.csv -m llm --max-rows 1000
```

#### 3. **Tesztelés kisebb adattal**
```bash
# Gyors teszt (1000 sor)
python main_windows.py -i data/data.csv -m ml --train --max-rows 1000
```

### Parancssori Paraméterek
| Paraméter | Rövid | Leírás | Alapértelmezett |
|-----------|-------|---------|----------------|
| `--input` | `-i` | Bemeneti CSV fájl elérési útja | (kötelező) |
| `--mode` | `-m` | Üzemmód: `ml` vagy `llm` | `ml` |
| `--train` | | Modell betanítása (csak ML módban) | `False` |
| `--max-rows` | | Feldolgozandó sorok maximális száma | `None` |

## 📊 Kimeneti Fájlok

A program futtatása után létrejönnek:

1. **`out_clean.csv`** - Tisztított adathalmaz
2. **`out_sessions.csv`** - Létrehozott session-ek időbeli korrelációval
3. **`out_detections.csv`** - Detektált fenyegetések előrejelzésekkel és magyarázatokkal
4. **`incidents_report.csv`** - Részletes incidensjelentés (csak magas bizalommal rendelkezők)

## 🧩 Fájlok Részletes Leírása

### **Agentek**

#### 1. `data_cleaner.py`
- **Cél**: Nyers adatok előfeldolgozása
- **Funkciók**: 
  - Oszlopátnevezés és normalizálás
  - Hiányzó értékek kezelése
  - Duplikátumok eltávolítása
  - Osztálykiegyensúlyozás (SMOTE)
  - Adatmennyiség korlátozása (`max_rows`)

#### 2. `correlator.py`
- **Cél**: Hálózati session-ek építése
- **Funkciók**:
  - 5 perces időablakokban történő csoportosítás
  - IP cím, port és protokoll alapú korreláció
  - Támadási minták felismerése
  - Kockázati pontszámítás

#### 3. `detector.py`
- **Cél**: Fenyegetések automatikus észlelése
- **Módok**:
  - **ML mód**: RandomForest modell betanítása/előrejelzése
  - **LLM mód**: Ollama LLM használata prompt-alapú detektáláshoz

#### 4. `explainer.py`
- **Cél**: Észlelések magyarázata
- **Módok**:
  - **Simple**: Feature importance vagy szabályalapú magyarázat (ajánlott)
  - **SHAP**: SHAP értékek alapján (speciális esetekre)
  - **LLM**: Természetes nyelvű magyarázat generálás

#### 5. `investigator.py`
- **Cél**: Incidensjelentések generálása
- **Funkciók**:
  - Bizalmi szint alapú szűrés
  - Incidensek csoportosítása
  - Részletes kártyák készítése

### **Segédfájlok**

#### `crew.py`
- Fő koordinátor, összeköti az összes ügynököt
- Progress bar-ok kezelése
- Hibakezelés és részleges eredmények visszaadása

#### `main_windows.py`
- Parancssori interfész
- CSV betöltés chunk-olvasással
- Pipeline indítása és eredmények mentése

#### `ml_helpers.py`
- Feature encoding (OneHotEncoder + StandardScaler)
- Datetime oszlopok automatikus kezelése
- Encoder mentése/betöltése

## ⚙️ Konfiguráció

### Crew inicializálás testreszabása (`main_windows.py`):
```python
crew = Crew(
    cleaner_kwargs={"max_rows": 50000},
    correlator_kwargs={
        "time_window": pd.Timedelta(minutes=5),
        "min_events": 3,
        "enable_attack_patterns": True
    },
    detector_kwargs={"mode": "ml"},
    explainer_kwargs={"mode": "simple"},  # simple, shap, vagy llm
    investigator_kwargs={
        "min_confidence": 0.1,
        "max_reports": 200
    },
)
```

### Magyarázó módok összehasonlítása:
| Mód | Sebesség | Pontosság | Ollama szükséges? | Ajánlás |
|-----|----------|-----------|-------------------|---------|
| **Simple** | ⚡ Nagyon gyors | ✅ Jó | ❌ Nem | Alapértelmezett |
| **SHAP** | 🐌 Lassú | ✅✅ Nagyon jó | ❌ Nem | Speciális elemzések |
| **LLM** | 🐢 Nagyon lassú | 🤔 Változó | ✅ Igen | Kísérleti |

## 🐛 Hibaelhárítás

### Gyakori problémák:

#### 1. **"SHAP kiszámítása sikertelen"**
- **Megoldás**: Válts "simple" módra
- **Módosítás**: `explainer_kwargs={"mode": "simple"}`

#### 2. **Memória túlcsordulás**
- **Megoldás**: Csökkentsd az adatmennyiséget
- **Parancs**: `--max-rows 10000`

#### 3. **Ollama modell nem található**
- **Megoldás 1**: Telepítsd a modellt: `ollama pull llama3`
- **Megoldás 2**: Használj más modellt: `llm_name="llama2"` a detector.py-ban
- **Megoldás 3**: Használd az ML módot (nem kell Ollama)

#### 4. **Futtatás túl lassú**
- **Optimalizálások**:
  ```bash
  # Csökkentsd a sorok számát
  python main_windows.py -i data.csv -m ml --max-rows 10000
  
  # Használd a simple magyarázó módot
  # (módosítsd a main_windows.py-t: "mode": "simple")
  ```

## 📈 Teljesítmény

### Becsült futási idők:
| Sorok száma | ML mód | LLM mód |
|-------------|--------|---------|
| 1,000 | ~10-30 másodperc | ~1-2 perc |
| 10,000 | ~1-3 perc | ~10-20 perc |
| 50,000 | ~5-10 perc | Nagyon lassú |
| 100,000 | ~15-25 perc | Nem ajánlott |


## 📚 Hasznos Linkek

- [Pandas dokumentáció](https://pandas.pydata.org/docs/)
- [Scikit-learn dokumentáció](https://scikit-learn.org/stable/)
- [Ollama modellek](https://ollama.ai/library)
- [SMOTE dokumentáció](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
