# agents/correlator.py
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.logger import get_logger
from tqdm import tqdm
import re

log = get_logger(__name__)

class Correlator:
    """
    Komplex időablak-alapú session építés és támadási láncok azonosítása.
    A PDF leírás alapján: "Events from the same IP within 5-minute windows"
    és "Create rules like: 'Failed auth + privilege escalation + lateral movement = potential breach'"
    """
    def __init__(self,
                 time_window: timedelta = timedelta(minutes=5),
                 min_events: int = 3,
                 enable_attack_patterns: bool = True):
        self.time_window = time_window
        self.min_events = min_events
        self.enable_attack_patterns = enable_attack_patterns
        
        # Támadási minták definiálása (a PDF alapján)
        self.attack_patterns = {
            'auth_attack': {
                'name': 'Hitelesítési támadás',
                'patterns': [
                    r'.*auth.*fail.*',
                    r'.*login.*fail.*',
                    r'.*password.*fail.*',
                    r'.*authentication.*fail.*'
                ],
                'score': 2
            },
            'privilege_escalation': {
                'name': 'Jogok emelése',
                'patterns': [
                    r'.*privilege.*',
                    r'.*escalation.*',
                    r'.*admin.*access.*',
                    r'.*root.*access.*'
                ],
                'score': 3
            },
            'lateral_movement': {
                'name': 'Oldalirányú mozgás',
                'patterns': [
                    r'.*lateral.*movement.*',
                    r'.*multiple.*hosts.*',
                    r'.*internal.*scan.*',
                    r'.*network.*discovery.*'
                ],
                'score': 2
            },
            'port_scan': {
                'name': 'Port szkennelés',
                'patterns': [
                    r'.*port.*scan.*',
                    r'.*multiple.*ports.*',
                    r'.*SYN.*flood.*',
                    r'.*connection.*attempt.*'
                ],
                'score': 1
            },
            'data_exfiltration': {
                'name': 'Adatkiszivárgás',
                'patterns': [
                    r'.*data.*exfil.*',
                    r'.*large.*transfer.*',
                    r'.*sensitive.*data.*',
                    r'.*encryption.*'
                ],
                'score': 3
            }
        }
        
        # Komplex szabályok (a PDF példája alapján)
        self.complex_rules = [
            {
                'name': 'potential_breach',
                'description': 'Failed auth + privilege escalation + lateral movement = potential breach',
                'required_patterns': ['auth_attack', 'privilege_escalation', 'lateral_movement'],
                'score': 10
            },
            {
                'name': 'reconnaissance_attack',
                'description': 'Port scan + multiple IPs = reconnaissance',
                'required_patterns': ['port_scan'],
                'min_unique_ips': 3,
                'score': 5
            },
            {
                'name': 'data_theft',
                'description': 'Privilege escalation + data exfiltration = data theft',
                'required_patterns': ['privilege_escalation', 'data_exfiltration'],
                'score': 8
            }
        ]

    @staticmethod
    def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
        """Időbélyeg oszlop konvertálása"""
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp") if 'timestamp' in df.columns else df

    def _detect_attack_patterns(self, events_text: list) -> dict:
        """Események szöveges leírásaiban támadási minták keresése"""
        detected_patterns = {}
        
        for pattern_name, pattern_info in self.attack_patterns.items():
            pattern_count = 0
            for event_text in events_text:
                if isinstance(event_text, str):
                    for regex_pattern in pattern_info['patterns']:
                        if re.search(regex_pattern, event_text, re.IGNORECASE):
                            pattern_count += 1
                            break  # Ha találtunk egyet ebben az eseményben, mehet a következőre
            
            if pattern_count > 0:
                detected_patterns[pattern_name] = {
                    'name': pattern_info['name'],
                    'count': pattern_count,
                    'score': pattern_info['score'] * pattern_count
                }
        
        return detected_patterns

    def _apply_complex_rules(self, session_data: dict, detected_patterns: dict) -> list:
        """Komplex szabályok alkalmazása a detektált mintákra"""
        matched_rules = []
        
        for rule in self.complex_rules:
            rule_matched = True
            
            # Ellenőrizzük a kötelező mintákat
            for required_pattern in rule.get('required_patterns', []):
                if required_pattern not in detected_patterns:
                    rule_matched = False
                    break
            
            # Ellenőrizzük az egyedi IP-k számát
            if rule_matched and 'min_unique_ips' in rule:
                unique_ips = len(session_data.get('src_ips', []))
                if unique_ips < rule['min_unique_ips']:
                    rule_matched = False
            
            # Ellenőrizzük az események számát
            if rule_matched and 'min_events' in rule:
                if session_data.get('n_events', 0) < rule['min_events']:
                    rule_matched = False
            
            if rule_matched:
                matched_rules.append({
                    'rule_name': rule['name'],
                    'description': rule['description'],
                    'score': rule['score'],
                    'matched_patterns': [detected_patterns[p]['name'] for p in rule.get('required_patterns', []) if p in detected_patterns]
                })
        
        return matched_rules

    def _analyze_session(self, events: pd.DataFrame) -> dict:
        """Egy session részletes elemzése"""
        # Alapvető információk
        first_event = events.iloc[0]
        src_ips = events['src_ip'].unique().tolist() if 'src_ip' in events.columns else []
        dst_ips = events['dst_ip'].unique().tolist() if 'dst_ip' in events.columns else []
        
        # Támadási minták keresése
        detected_patterns = {}
        if self.enable_attack_patterns:
            # Összegyűjtjük az események leírásait
            event_descriptions = []
            
            # Próbáljuk megtalálni a leíró oszlopokat
            description_cols = []
            for col in events.columns:
                col_lower = col.lower()
                if ('label' in col_lower or 'description' in col_lower or 
                    'type' in col_lower or 'name' in col_lower):
                    description_cols.append(col)
            
            if description_cols:
                for col in description_cols:
                    event_descriptions.extend(events[col].astype(str).tolist())
            else:
                # Ha nincs leíró oszlop, használjuk az összes szöveges oszlopot
                text_cols = events.select_dtypes(include=['object']).columns
                for col in text_cols:
                    event_descriptions.extend(events[col].astype(str).tolist())
            
            if event_descriptions:
                detected_patterns = self._detect_attack_patterns(event_descriptions)
        
        # Komplex szabályok alkalmazása
        matched_rules = []
        if detected_patterns:
            session_info = {
                'src_ips': src_ips,
                'dst_ips': dst_ips,
                'n_events': len(events)
            }
            matched_rules = self._apply_complex_rules(session_info, detected_patterns)
        
        # Kockázati pontszám számítása
        risk_score = 0
        for pattern_info in detected_patterns.values():
            risk_score += pattern_info['score']
        
        for rule in matched_rules:
            risk_score += rule['score']
        
        # Támadási minták szöveggé alakítása (lista helyett)
        detected_patterns_str = ""
        if detected_patterns:
            pattern_list = []
            for pattern_name, pattern_info in detected_patterns.items():
                pattern_list.append(f"{pattern_info['name']}({pattern_info['count']})")
            detected_patterns_str = "; ".join(pattern_list)
        
        # Egyeztetett szabályok szöveggé alakítása
        matched_rules_str = ""
        if matched_rules:
            rule_list = []
            for rule in matched_rules:
                rule_list.append(f"{rule['rule_name']}")
            matched_rules_str = "; ".join(rule_list)
        
        # Támadási lánc leírás
        attack_description = self._generate_attack_description(detected_patterns, matched_rules)
        
        # Alapvető oszlopok biztosítása
        result = {
            'session_id': f"{first_event['src_ip']}_{first_event['dst_ip']}_{int(events['timestamp'].iloc[0].timestamp())}" if 'timestamp' in events.columns else f"{first_event['src_ip']}_{first_event['dst_ip']}",
            'src_ip': first_event.get('src_ip', 'unknown'),
            'dst_ip': first_event.get('dst_ip', 'unknown'),
            'src_port': first_event.get('src_port', 0),
            'dst_port': first_event.get('dst_port', 0),
            'protocol': first_event.get('protocol', 'unknown'),
            'start_time': events['timestamp'].min() if 'timestamp' in events.columns else pd.Timestamp.now(),
            'end_time': events['timestamp'].max() if 'timestamp' in events.columns else pd.Timestamp.now(),
            'duration_sec': (events['timestamp'].max() - events['timestamp'].min()).total_seconds() if 'timestamp' in events.columns else 0,
            'n_events': len(events),
            'unique_src_ips': len(src_ips),
            'unique_dst_ips': len(dst_ips),
            'label_majority': events['label'].mode().iloc[0] if 'label' in events.columns else 'unknown',
            'detected_patterns': detected_patterns_str,  # Szöveg, nem lista!
            'matched_rules': matched_rules_str,  # Szöveg, nem lista!
            'risk_score': risk_score,
            'attack_chain_description': attack_description
        }
        
        return result

    def _generate_attack_description(self, detected_patterns: dict, matched_rules: list) -> str:
        """Támadási leírás generálása"""
        descriptions = []
        
        # Alap minták leírása
        for pattern_name, pattern_info in detected_patterns.items():
            desc = f"{pattern_info['name']} ({pattern_info['count']} esemény)"
            descriptions.append(desc)
        
        # Komplex szabályok leírása
        for rule in matched_rules:
            desc = f"Szabály: {rule['description']}"
            descriptions.append(desc)
        
        if descriptions:
            return "; ".join(descriptions)
        else:
            return "Normális tevékenység"

    def _group_sessions_with_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Session-ek csoportosítása komplex szabályokkal"""
        df = self._ensure_ts(df)
        
        # Ellenőrizzük a kötelező oszlopokat
        if 'src_ip' not in df.columns:
            log.error("Hiányzó 'src_ip' oszlop!")
            return pd.DataFrame()
        
        if 'timestamp' not in df.columns:
            # Próbáljuk megkeresni időbélyeg oszlopot
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df = df.rename(columns={time_cols[0]: 'timestamp'})
                df = self._ensure_ts(df)
            else:
                log.error("Hiányzó időbélyeg oszlop!")
                return pd.DataFrame()
        
        # 1. Forrás IP alapú csoportosítás (a PDF alapján: "Events from the same IP")
        sessions = []
        
        # Egyedi forrás IP-k
        unique_src_ips = df['src_ip'].unique()
        log.info(f"Session építés {len(unique_src_ips)} egyedi forrás IP-hez...")
        
        # Progress bar
        pbar = tqdm(
            total=len(unique_src_ips), 
            desc="Korreláció elemzés",
            unit="IP",
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='blue'
        )
        
        for src_ip in unique_src_ips:
            # Az adott IP összes eseménye időrendben
            ip_events = df[df['src_ip'] == src_ip].sort_values('timestamp')
            
            if len(ip_events) >= self.min_events:
                current_window = []
                current_start = ip_events.iloc[0]['timestamp']
                
                for _, event in ip_events.iterrows():
                    # 5 perces időablak ellenőrzése (a PDF alapján)
                    if (event['timestamp'] - current_start) <= self.time_window:
                        current_window.append(event)
                    else:
                        # Új időablak kezdődik
                        if len(current_window) >= self.min_events:
                            session_df = pd.DataFrame(current_window)
                            session_info = self._analyze_session(session_df)
                            sessions.append(session_info)
                        
                        current_window = [event]
                        current_start = event['timestamp']
                
                # Utolsó időablak feldolgozása
                if len(current_window) >= self.min_events:
                    session_df = pd.DataFrame(current_window)
                    session_info = self._analyze_session(session_df)
                    sessions.append(session_info)
            
            pbar.update(1)
            if len(sessions) > 0:
                pbar.set_postfix({
                    "session-ek": len(sessions),
                    "kockázat átlag": f"{np.mean([s.get('risk_score', 0) for s in sessions[-10:]]) if len(sessions) >= 10 else 0:.1f}"
                })
        
        pbar.close()
        
        if sessions:
            sess_df = pd.DataFrame(sessions)
            
            # Rendezés kockázati pontszám szerint
            if 'risk_score' in sess_df.columns:
                sess_df = sess_df.sort_values('risk_score', ascending=False)
            
            log.info(f"Talált session-ek: {len(sess_df)}")
            
            # Statisztikák
            if 'detected_patterns' in sess_df.columns and len(sess_df) > 0:
                # Számláljuk a detektált mintákat
                all_patterns = []
                for patterns in sess_df['detected_patterns']:
                    if isinstance(patterns, str) and patterns:
                        all_patterns.extend([p.strip() for p in patterns.split(';')])
                
                from collections import Counter
                pattern_counts = Counter(all_patterns)
                
                if pattern_counts:
                    log.info("Detektált támadási minták (top 5):")
                    for pattern, count in pattern_counts.most_common(5):
                        log.info(f"  {pattern}: {count}")
            
            if 'matched_rules' in sess_df.columns and len(sess_df) > 0:
                # Számláljuk a szabályokat
                all_rules = []
                for rules in sess_df['matched_rules']:
                    if isinstance(rules, str) and rules:
                        all_rules.extend([r.strip() for r in rules.split(';')])
                
                rule_counts = Counter(all_rules)
                
                if rule_counts:
                    log.info("Egyeztetett komplex szabályok:")
                    for rule, count in rule_counts.most_common():
                        rule_desc = next((r['description'] for r in self.complex_rules if r['name'] == rule), rule)
                        log.info(f"  {rule}: {count} alkalom")
        else:
            sess_df = pd.DataFrame()
            log.warning("Nem található session a megadott feltételekkel!")
        
        return sess_df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fő futtató metódus"""
        log.info("Korreláció indítása - Támadási láncok és komplex szabályok keresése")
        
        # Ellenőrizzük, hogy van-e szükséges oszlop
        required_cols = ['src_ip', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log.warning(f"Hiányzó oszlopok: {missing_cols}")
            
            # Próbáljuk megkeresni hasonló oszlopokat
            for missing_col in missing_cols:
                possible_matches = [col for col in df.columns if missing_col in col.lower()]
                if possible_matches:
                    log.info(f"  Átnevezés: {possible_matches[0]} → {missing_col}")
                    df = df.rename(columns={possible_matches[0]: missing_col})
        
        return self._group_sessions_with_rules(df)