"""
Temporal Knowledge Grounding System for AI Search.
Detects when queries need real-time data vs. knowledge base answers.
"""

import re
import random
from typing import List, Optional, Dict
import numpy as np

# CONFIGURATION

MARKERS = {
    'current': {
        'patterns': [r'\bnow\b', r'\btoday\b', r'\bright\b'],
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
        'patterns': [r'\bwas\b', r'\bwere\b', r'\bancient\b', r'\bhistorically\b', r'\binvented\b'],
        'boost': -0.3
    },
}

DOMAINS = {
    'finance': {
        'keywords': ['stock', 'market', 'price', 'trading', 'investment', 'crypto', 'bitcoin', 'finance'],
        'volatility': 0.8,
        'subdomains': {
            'crypto': {'keywords': ['bitcoin', 'ethereum', 'crypto', 'defi'], 'vol': 1.0},
            'stocks': {'keywords': ['stock', 'shares', 'nasdaq', 'nyse', 'dow'], 'vol': 1.0},
            'forex': {'keywords': ['forex', 'currency', 'exchange rate', 'dollar', 'euro'], 'vol': 1.0},
        }
    },
    'news': {
        'keywords': ['news', 'breaking', 'happening', 'latest'],
        'volatility': 0.9,
        'subdomains': {
            'politics': {'keywords': ['election', 'president', 'government', 'policy'], 'vol': 0.7},
            'sports': {'keywords': ['score', 'game', 'match', 'win', 'player', 'team'], 'vol': 0.8},
        }
    },
    'technology': {
        'keywords': ['tech', 'software', 'app', 'programming', 'ai', 'ml', 'python', 'javascript', 'developer'],
        'volatility': 0.5,
        'subdomains': {
            'ai_research': {'keywords': ['ai', 'ml', 'llm', 'gpt', 'bert', 'model', 'paper', 'research'], 'vol': 0.7},
            'products': {'keywords': ['iphone', 'android', 'release', 'launch', 'announce'], 'vol': 0.7},
        }
    },
    'science': {
        'keywords': ['science', 'research', 'physics', 'biology', 'chemistry'],
        'volatility': 0.4,
        'subdomains': {
            'constants': {'keywords': ['speed of light', 'gravity', 'constant', 'atom'], 'vol': 0.0},
            'medical': {'keywords': ['treatment', 'drug', 'vaccine', 'fda', 'medical'], 'vol': 0.5},
        }
    },
    'history': {
        'keywords': ['history', 'historical', 'ancient', 'war', 'century', 'era'],
        'volatility': 0.0
    },
    'geography': {
        'keywords': ['country', 'city', 'capital', 'continent', 'population', 'located'],
        'volatility': 0.05
    },
}


# MAIN SYSTEM

class TemporalGroundingSystem:
    """
    Detects temporal urgency in queries to determine when AI needs fresh data.
    
    Score interpretation:
    - 0.0-0.3: Stable knowledge (historical facts, definitions)
    - 0.3-0.5: Generally stable, verification helpful
    - 0.5-0.7: Needs recent data
    - 0.7-1.0: Critical need for real-time data
    """
    
    def __init__(self):
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self) -> Dict:
        """Build keyword to (domain, subdomain) mapping."""
        mapping = {}
        for domain, config in DOMAINS.items():
            for kw in config.get('keywords', []):
                mapping[kw] = (domain, None)
            for sub, subcfg in config.get('subdomains', {}).items():
                for kw in subcfg.get('keywords', []):
                    mapping[kw] = (domain, sub)
        return mapping
    
    def analyze(self, query: str) -> dict:
        """
        Analyze a query and return temporal grounding assessment.
        
        Returns:
            dict with:
            - query: original query
            - score: freshness score (0-1)
            - action: 'retrieve_fresh' or 'use_knowledge_base'
            - domain: identified domain
            - markers: detected temporal markers
        """
        query_lower = query.lower()
        
        # Get domain volatility (0-1 scale)
        vol = self._get_volatility(query_lower)
        
        # Get temporal score (0-1 scale)
        temporal_score = self._get_temporal_score(query_lower)
        
        # Get detected markers
        markers = self._detect_markers(query_lower)
        
        # Combine based on whether markers exist
        if markers:
            # Has temporal markers → trust them more
            combined = 0.6 * temporal_score + 0.4 * vol
        else:
            # No markers → trust domain volatility more
            combined = 0.3 * temporal_score + 0.7 * vol
        
        # Clamp to 0-1
        combined = max(0.0, min(1.0, combined))
        
        # Decision
        action = 'retrieve_fresh' if combined >= 0.5 else 'use_knowledge_base'
        
        return {
            'query': query,
            'score': round(combined, 2),
            'action': action,
            'domain': self._predict_domain(query_lower),
            'markers': markers,
        }
    
    def _get_temporal_score(self, query: str) -> float:
        """
        Convert detected markers to a 0-1 freshness score.
        Historical markers → low score
        Recent markers → high score
        """
        markers = self._detect_markers(query)
        
        if not markers:
            return 0.5  # Neutral
        
        # Sum all marker boosts
        total_boost = sum(MARKERS[m]['boost'] for m in markers)
        
        # Map to 0-1 scale:
        # -0.3 (all historical) → 0.2
        # 0.0 (neutral) → 0.5
        # +0.4 (current) → 0.9
        score = 0.5 + total_boost
        
        return max(0.0, min(1.0, score))
    
    def _detect_markers(self, query: str) -> List[str]:
        """Detect all temporal markers in the query."""
        found = []
        for marker, config in MARKERS.items():
            if any(re.search(p, query) for p in config['patterns']):
                found.append(marker)
        return found
    
    def _predict_domain(self, query: str) -> Optional[str]:
        """Predict the primary domain of the query."""
        scores = {}
        for kw, (domain, _) in self.keyword_map.items():
            if kw in query:
                scores[domain] = scores.get(domain, 0) + 1
        
        if not scores:
            return None
        
        return max(scores, key=scores.get)
    
    def _get_volatility(self, query: str) -> float:
        """
        Get volatility score based on domain/subdomain.
        Returns 0.0 (never changes) to 1.0 (real-time changes)
        """
        # Check subdomain first (more specific)
        for kw, (domain, sub) in self.keyword_map.items():
            if kw in query and sub:
                return DOMAINS[domain]['subdomains'][sub]['vol']
        
        # Fall back to domain
        for kw, (domain, _) in self.keyword_map.items():
            if kw in query:
                return DOMAINS[domain]['volatility']
        
        # Default: medium volatility
        return 0.5

# EVALUATION DATASET

EVALUATION_DATASET = [
    # FINANCE - High freshness need
    {"query": "What is Bitcoin trading at right now?", "freshness": 1.0},
    {"query": "What is the current stock price of Tesla?", "freshness": 1.0},
    {"query": "What is the EUR to USD rate today?", "freshness": 1.0},
    {"query": "What are today's gas prices?", "freshness": 1.0},
    {"query": "What stocks are trending today?", "freshness": 0.95},
    {"query": "What is the current inflation rate?", "freshness": 0.8},
    {"query": "What companies are in the S&P 500?", "freshness": 0.7},
    
    # NEWS - High freshness need
    {"query": "What breaking news today?", "freshness": 1.0},
    {"query": "What happened in Ukraine recently?", "freshness": 1.0},
    {"query": "What is the latest election news?", "freshness": 0.95},
    {"query": "What sports results today?", "freshness": 0.95},
    {"query": "What is trending on Twitter?", "freshness": 1.0},
    {"query": "What political scandal is in the news?", "freshness": 0.9},
    
    # TECHNOLOGY - Medium freshness need
    {"query": "What is the latest AI model?", "freshness": 0.9},
    {"query": "What new iPhone features announced?", "freshness": 0.85},
    {"query": "What is the newest React version?", "freshness": 0.85},
    {"query": "What is the current Python version?", "freshness": 0.8},
    {"query": "What is the GPT-5 release date?", "freshness": 0.85},
    {"query": "What is the latest JavaScript framework?", "freshness": 0.75},
    {"query": "How does blockchain work?", "freshness": 0.4},
    {"query": "What is machine learning?", "freshness": 0.2},
    
    # SCIENCE - Variable freshness
    {"query": "What is the speed of light?", "freshness": 0.0},
    {"query": "What is quantum entanglement?", "freshness": 0.2},
    {"query": "What is the treatment for diabetes?", "freshness": 0.5},
    {"query": "How do vaccines work?", "freshness": 0.2},
    {"query": "What did the latest research paper say?", "freshness": 0.8},
    
    # STATIC - Low freshness need
    {"query": "What is the capital of France?", "freshness": 0.05},
    {"query": "Who was the first president?", "freshness": 0.0},
    {"query": "What is the theory of relativity?", "freshness": 0.05},
    {"query": "Who invented the printing press?", "freshness": 0.0},
    {"query": "What are the three states of matter?", "freshness": 0.0},
    {"query": "How many continents are there?", "freshness": 0.0},
    
    # MIXED
    {"query": "Who is the current CEO of Microsoft?", "freshness": 0.6},
    {"query": "What are the current COVID-19 guidelines?", "freshness": 0.7},
    {"query": "What is the best Python framework?", "freshness": 0.5},
    {"query": "What is the weather in London?", "freshness": 1.0},
    {"query": "What are the current health guidelines?", "freshness": 0.7},
    {"query": "What programming language should I learn?", "freshness": 0.55},
]


# BASELINES

def run_baselines(data: List[dict]) -> dict:
    """Compare system against simple baselines."""
    
    # Random baseline
    random.seed(42)
    random_preds = [random.random() for _ in data]
    random_mae = np.mean([abs(p - d['freshness']) for p, d in zip(random_preds, data)])
    
    # Keyword-only baseline
    keyword_preds = []
    for d in data:
        q = d['query'].lower()
        if any(kw in q for kw in ['current', 'latest', 'now', 'today', 'right']):
            keyword_preds.append(0.8)
        elif any(kw in q for kw in ['was', 'ancient', 'invented']):
            keyword_preds.append(0.1)
        else:
            keyword_preds.append(0.5)
    keyword_mae = np.mean([abs(p - d['freshness']) for p, d in zip(keyword_preds, data)])
    
    return {
        'random': random_mae,
        'keyword': keyword_mae,
    }


# MAIN EVALUATION

def evaluate():
    """Run complete evaluation."""
    system = TemporalGroundingSystem()
    
    # Get predictions
    predictions = [system.analyze(d['query'])['score'] for d in EVALUATION_DATASET]
    
    # Calculate MAE
    maes = [abs(p - d['freshness']) for p, d in zip(predictions, EVALUATION_DATASET)]
    mae = np.mean(maes)
    
    # Baselines
    baselines = run_baselines(EVALUATION_DATASET)
    
    # Binary classification
    pred_binary = ['retrieve_fresh' if p >= 0.5 else 'use_knowledge_base' for p in predictions]
    gt_binary = ['retrieve_fresh' if d['freshness'] >= 0.5 else 'use_knowledge_base' for d in EVALUATION_DATASET]
    
    accuracy = sum(1 for p, g in zip(pred_binary, gt_binary) if p == g) / len(pred_binary)
    
    # F1
    tp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 'retrieve_fresh' and g == 'retrieve_fresh')
    fp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 'retrieve_fresh' and g == 'use_knowledge_base')
    fn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 'use_knowledge_base' and g == 'retrieve_fresh')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("=" * 60)
    print("TEMPORAL KNOWLEDGE GROUNDING SYSTEM - EVALUATION")
    print("=" * 60)
    print(f"\nDataset: {len(EVALUATION_DATASET)} queries")
    print(f"\nBASELINES:")
    print(f"  Random (0-1):     MAE {baselines['random']:.4f}")
    print(f"  Keyword Only:      MAE {baselines['keyword']:.4f}")
    print(f"\nOUR SYSTEM:")
    print(f"  MAE:               {mae:.4f}")
    print(f"  vs Random:         {(baselines['random'] - mae) / baselines['random'] * 100:.1f}% better")
    print(f"  vs Keyword:        {(baselines['keyword'] - mae) / baselines['keyword'] * 100:.1f}% better")
    print(f"\nCLASSIFICATION:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  F1 Score: {f1:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    
    return mae, accuracy, f1

# DEMO

def demo():
    """Demo with sample queries."""
    system = TemporalGroundingSystem()
    
    queries = [
        "What is Bitcoin trading at right now?",
        "What is the capital of France?",
        "What are the latest developments in AI?",
        "Who invented the printing press?",
        "What is the current CEO of Microsoft?",
    ]
    
    print("\nDEMO:")
    print("-" * 60)
    for query in queries:
        result = system.analyze(query)
        print(f"\nQuery: {query}")
        print(f"  Score: {result['score']} | Action: {result['action']}")
        print(f"  Domain: {result['domain']} | Markers: {result['markers']}")


if __name__ == "__main__":
    evaluate()
    demo()

