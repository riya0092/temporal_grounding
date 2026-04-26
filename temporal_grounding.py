# temporal_grounding.py

"""
Temporal Knowledge Grounding System for AI Search.
Detects when queries need real-time data vs. knowledge base answers.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# MARKER PATTERNS

MARKERS = {
    'current': {
        'patterns': [r'\bnow\b', r'\btoday\b', r'\bright (now|now)\b'],
        'boost': 0.4
    },
    'recent': {
        'patterns': [r'\blastest\b', r'\bnewest\b', r'\bmost recent\b', r'\brecently\b'],
        'boost': 0.35
    },
    'future': {
        'patterns': [r'\bwill\b', r'\bgoing to\b', r'\bpredicted\b'],
        'boost': 0.2
    },
    'time_bound': {
        'patterns': [r'\b(202|201)\d\b', r'\bthis (week|month|year)\b'],
        'boost': 0.25
    },
    'historical': {
        'patterns': [r'\bwas\b', r'\bwere\b', r'\bancient\b', r'\bhistorically\b'],
        'boost': -0.3
    },
}


# DOMAIN VOLATILITY

DOMAINS = {
    'finance': {
        'keywords': ['stock', 'market', 'price', 'trading', 'investment', 'crypto', 'bitcoin'],
        'volatility': 1.0,
        'subdomains': {
            'crypto': {'keywords': ['bitcoin', 'ethereum', 'crypto'], 'vol': 1.0},
            'stocks': {'keywords': ['stock', 'shares', 'nasdaq', 'nyse'], 'vol': 1.0},
        }
    },
    'news': {
        'keywords': ['news', 'breaking', 'happening'],
        'volatility': 1.0,
        'subdomains': {
            'politics': {'keywords': ['election', 'president', 'government'], 'vol': 0.8},
            'sports': {'keywords': ['score', 'game', 'match', 'player'], 'vol': 0.8},
        }
    },
    'technology': {
        'keywords': ['tech', 'software', 'app', 'programming', 'ai', 'ml'],
        'volatility': 0.5,
        'subdomains': {
            'ai_research': {'keywords': ['ai', 'ml', 'llm', 'gpt', 'model', 'paper'], 'vol': 0.8},
            'products': {'keywords': ['iphone', 'android', 'release', 'launch'], 'vol': 0.8},
        }
    },
    'science': {
        'keywords': ['science', 'research', 'physics', 'biology'],
        'volatility': 0.5,
        'subdomains': {
            'constants': {'keywords': ['speed of light', 'gravity', 'constant'], 'vol': 0.0},
        }
    },
    'history': {
        'keywords': ['history', 'historical', 'ancient', 'war', 'century'],
        'volatility': 0.0
    },
    'geography': {
        'keywords': ['country', 'city', 'capital', 'continent'],
        'volatility': 0.0
    },
}


# CORE CLASSES

class TemporalGroundingSystem:
    """Detects temporal urgency in queries."""
    
    KNOWLEDGE_CUTOFF = datetime(2023, 9, 1)
    
    def __init__(self):
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self) -> Dict:
        mapping = {}
        for domain, config in DOMAINS.items():
            for kw in config.get('keywords', []):
                mapping[kw] = (domain, None)
            for sub, subcfg in config.get('subdomains', {}).items():
                for kw in subcfg.get('keywords', []):
                    mapping[kw] = (domain, sub)
        return mapping
    
    def analyze(self, query: str) -> dict:
        query_lower = query.lower()
        
        # Detect markers
        markers = self._detect_markers(query_lower)
        marker_score = sum(MARKERS[m]['boost'] for m in markers) / max(1, len(markers))
        
        # Predict domain
        domain, subdomain = self._predict_domain(query_lower)
        vol = self._get_volatility(query_lower)
        
        # Combine scores
        weight_temporal = 0.6 if markers else 0.2
        weight_domain = 0.4 if not markers else 0.8
        combined = weight_temporal * (0.3 + marker_score) + weight_domain * vol
        combined = min(1.0, max(0.0, combined))
        
        # Decision
        if combined >= 0.7 or vol >= 1.0:
            status = 'critically_stale'
            action = 'retrieve_fresh'
        elif combined >= 0.5:
            status = 'likely_stale'
            action = 'retrieve_fresh'
        elif combined >= 0.3:
            status = 'possibly_stale'
            action = 'verify'
        else:
            status = 'up_to_date'
            action = 'use_knowledge_base'
        
        return {
            'query': query,
            'score': round(combined, 2),
            'status': status,
            'action': action,
            'domain': domain,
            'markers': markers,
            'confidence': round(max(0.5, 1 - combined * 0.4), 2)
        }
    
    def _detect_markers(self, query: str) -> List[str]:
        found = []
        for marker, config in MARKERS.items():
            if any(re.search(p, query) for p in config['patterns']):
                found.append(marker)
        return found
    
    def _predict_domain(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        scores = {}
        for kw, (domain, _) in self.keyword_map.items():
            if kw in query:
                scores[domain] = scores.get(domain, 0) + 1
        if not scores:
            return None, None
        return max(scores, key=scores.get), None
    
    def _get_volatility(self, query: str) -> float:
        for kw, (domain, sub) in self.keyword_map.items():
            if kw in query and sub:
                return DOMAINS[domain]['subdomains'][sub]['vol']
        for kw, (domain, _) in self.keyword_map.items():
            if kw in query:
                return DOMAINS[domain]['volatility']
        return 0.5  # default


# EVALUATION

EVALUATION_DATA = [
    {"q": "What is Bitcoin trading at right now?", "gt": 1.0, "ret": True},
    {"q": "What is the current stock price of Tesla?", "gt": 1.0, "ret": True},
    {"q": "What breaking news is happening?", "gt": 1.0, "ret": True},
    {"q": "Who won the latest Premier League match?", "gt": 0.9, "ret": True},
    {"q": "What are the latest developments in AI?", "gt": 0.9, "ret": True},
    {"q": "What new iPhone features were announced?", "gt": 0.85, "ret": True},
    {"q": "What is the current unemployment rate?", "gt": 0.8, "ret": True},
    {"q": "What is the capital of France?", "gt": 0.05, "ret": False},
    {"q": "What is the theory of relativity?", "gt": 0.05, "ret": False},
    {"q": "Who invented the printing press?", "gt": 0.0, "ret": False},
    {"q": "What are the three states of matter?", "gt": 0.0, "ret": False},
    {"q": "How does blockchain work?", "gt": 0.4, "ret": False},
    {"q": "Who is the current CEO of Microsoft?", "gt": 0.6, "ret": True},
    {"q": "What is the latest Python version?", "gt": 0.8, "ret": True},
]


def evaluate():
    """Run evaluation on test data."""
    system = TemporalGroundingSystem()
    
    results = []
    for item in EVALUATION_DATA:
        pred = system.analyze(item['q'])
        results.append({
            'query': item['q'],
            'ground_truth': item['gt'],
            'predicted': pred['score'],
            'needs_retrieval_gt': item['ret'],
            'needs_retrieval_pred': pred['action'] == 'retrieve_fresh'
        })
    
    df = pd.DataFrame(results)
    mae = np.mean(np.abs(df['ground_truth'] - df['predicted']))
    accuracy = np.mean(df['needs_retrieval_gt'] == df['needs_retrieval_pred'])
    
    print(f"MAE: {mae:.3f}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"\nSample predictions:")
    for _, row in df.sample(5).iterrows():
        print(f"  [{row['ground_truth']:.1f}→{row['predicted']:.2f}] {row['query'][:50]}...")
    
    return df


if __name__ == "__main__":
    print("Temporal Knowledge Grounding System")
    print("=" * 40)
    evaluate()
