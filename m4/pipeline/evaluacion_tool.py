# -*- coding: utf-8 -*-
"""
Métricas del componente visual (M4). Solo numpy.

Convención: y = 1 es urgente. Un score es P(urgente) entre 0 y 1.

Decisión sobre el estado `no_evaluable` (la válvula de la tool): para las métricas
ESTRICTAS cuenta como "no dijo urgente", es decir, un urgente marcado no_evaluable
se cuenta como no detectado. Se reporta aparte `recall_con_escalamiento`, donde
no_evaluable cuenta como detectado porque el sistema lo manda a revisión humana.
Es la misma separación que hicimos en M3 con la válvula del RAG.
"""

import math

import numpy as np


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción k/n.

    Con n pequeño (20 urgentes en el eval set de la tool) el intervalo es ancho,
    y eso es justo lo que hay que mostrar junto al punto estimado.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centro = p + z * z / (2 * n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centro - margen) / d, (centro + margen) / d)


def auc(y, scores):
    """AUC por comparación de pares (empates cuentan 0.5)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def umbral_para_recall(y, scores, recall_min=0.85):
    """Umbral MÁS ALTO (o sea, de mayor especificidad) con recall >= recall_min.

    Se predice urgente cuando score >= umbral. Es exacto: el umbral es el
    k-ésimo mayor score entre los urgentes, con k = ceil(recall_min * n_urgentes).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = np.sort(s[y == 1])[::-1]
    neg = s[y == 0]
    if len(pos) == 0:
        raise ValueError("No hay casos urgentes para calibrar el umbral")
    k = int(math.ceil(round(recall_min * len(pos), 9)))
    k = max(1, min(k, len(pos)))
    umbral = float(pos[k - 1])
    return {
        "umbral": umbral,
        "recall": float((pos >= umbral).mean()),
        "especificidad": float((neg < umbral).mean()) if len(neg) else float("nan"),
    }


def resumen_tool(esperado, clase):
    """Resumen sobre listas de strings.

    esperado: 'urgente' | 'no_urgente'
    clase:    'urgente' | 'no_urgente' | 'no_evaluable'
    """
    esperado = list(esperado)
    clase = list(clase)
    if len(esperado) != len(clase):
        raise ValueError("esperado y clase deben tener el mismo largo")
    n = len(esperado)
    urg = [c for e, c in zip(esperado, clase) if e == "urgente"]
    nou = [c for e, c in zip(esperado, clase) if e == "no_urgente"]
    tp = sum(c == "urgente" for c in urg)
    tn = sum(c == "no_urgente" for c in nou)
    fn = len(urg) - tp
    fp = sum(c == "urgente" for c in nou)
    escalados = sum(c == "no_evaluable" for c in urg)
    correctos = tp + tn
    return {
        "n": n,
        "n_urgentes": len(urg),
        "n_no_urgentes": len(nou),
        "verdaderos_positivos": tp,
        "falsos_negativos": fn,
        "falsos_positivos": fp,
        "verdaderos_negativos": tn,
        "no_evaluables_total": sum(c == "no_evaluable" for c in clase),
        "accuracy": correctos / n if n else float("nan"),
        "accuracy_ic95": wilson(correctos, n),
        "recall_urgente": tp / len(urg) if urg else float("nan"),
        "recall_urgente_ic95": wilson(tp, len(urg)),
        "recall_con_escalamiento": (tp + escalados) / len(urg) if urg else float("nan"),
        "especificidad": tn / len(nou) if nou else float("nan"),
        "especificidad_ic95": wilson(tn, len(nou)),
    }
