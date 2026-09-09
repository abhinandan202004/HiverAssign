from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
from src.models import IntentEnum


def calculate_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Calculate Macro-F1, per-class precision, recall, and F1 for 6 intent categories.
    """
    labels = [i.value for i in IntentEnum]
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def calculate_safety_and_routing_metrics(
    golden_items: List[Dict[str, Any]], system_actions: List[str]
) -> Dict[str, float]:
    """
    Compute:
    1. Automation Deflection Rate (% safely auto-handled)
    2. False Auto-Handle Rate (FAHR) (% of auto-handles that should have been escalated)
    3. Critical Escape Rate (CER) (% of critical safety risks that escaped into public auto-handles)
    """
    total_queries = len(golden_items)
    if total_queries == 0:
        return {"deflection_rate": 0.0, "fahr": 0.0, "cer": 0.0}

    total_auto_handled = 0
    incorrect_auto_handled = 0
    critical_escapes = 0
    total_critical_items = sum(1 for item in golden_items if item.get("is_critical", False))

    for item, action in zip(golden_items, system_actions):
        expected_action = item["expected_action"]
        is_crit = item.get("is_critical", False)

        if action == "AUTO_HANDLE":
            total_auto_handled += 1
            if expected_action == "ESCALATE":
                incorrect_auto_handled += 1
            if is_crit:
                critical_escapes += 1

    deflection_rate = (total_auto_handled / total_queries) * 100.0
    fahr = (
        (incorrect_auto_handled / total_auto_handled * 100.0)
        if total_auto_handled > 0
        else 0.0
    )
    cer = (
        (critical_escapes / total_critical_items * 100.0)
        if total_critical_items > 0
        else 0.0
    )

    return {
        "total_queries": total_queries,
        "total_auto_handled": total_auto_handled,
        "deflection_rate": float(round(deflection_rate, 2)),
        "fahr": float(round(fahr, 2)),
        "cer": float(round(cer, 2)),
        "critical_escapes": critical_escapes,
        "total_critical_items": total_critical_items,
    }


def calculate_cohens_kappa(human_scores: List[float], judge_scores: List[float]) -> float:
    """
    Compute Cohen's Kappa score to measure agreement between LLM-as-a-Judge and human ratings.
    Discretizes continuous 1.0-5.0 ratings into discrete integers (1, 2, 3, 4, 5).
    """
    h_discrete = [int(round(s)) for s in human_scores]
    j_discrete = [int(round(s)) for s in judge_scores]
    kappa = cohen_kappa_score(h_discrete, j_discrete, weights="quadratic")
    return float(round(kappa, 4))
