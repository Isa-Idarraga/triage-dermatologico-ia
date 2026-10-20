"""Regresiones del cierre con el agente completo y dobles de modelos/herramientas."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import decision_m3 as dec


def bloqueado(*args, **kwargs):
    raise AssertionError("Esta suite no debe cargar pesos ni usar servicios")


def cargar_agente():
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=bloqueado)
    ra = types.ModuleType("rag_avanzado")
    ra.indice = ra.sistema = bloqueado
    mm = types.ModuleType("metricas_m2")
    mm.sistema = bloqueado
    with patch.dict(sys.modules, {"transformers": tf, "rag_avanzado": ra, "metricas_m2": mm}):
        spec = importlib.util.spec_from_file_location("agente_en_prueba", ROOT / "scripts/agente_tools.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


A = cargar_agente()
B, R = "clasificador_beto", dec.RAG


def usar(nombre):
    return {"accion": "usar_herramienta", "herramienta": nombre}


def cerrar(texto="URGENCIA: no_urgente"):
    return {"accion": "responder_final", "respuesta": texto}


def ejecutar(decisiones, beto=None, rag=None, fallas=()):
    callbacks = {B: Mock(return_value=beto or {"etiqueta": "urgente", "confianza": .95}),
                 R: Mock(return_value=rag or {"etiqueta": "no_urgente", "respuesta": "URGENCIA: no_urgente",
                                              "contexto": ["dato de prueba"], "fuentes": ["f1", "f1"]})}
    for nombre in fallas:
        callbacks[nombre].side_effect = RuntimeError("fallo controlado")
    herramientas = {k: {"funcion": fn, "descripcion": "prueba", "cuando_usarla": "prueba"}
                    for k, fn in callbacks.items()}
    pasos = [(d, "salida controlada") for d in decisiones]
    with patch.object(A, "HERRAMIENTAS", herramientas), patch.object(A, "_preguntar_al_cerebro", side_effect=pasos):
        return A.agente("consulta sintética"), callbacks


class ParserTests(unittest.TestCase):
    def test_valid_tool(self):
        self.assertEqual(A._parsear_decision('{"accion":"usar_herramienta","herramienta":"clasificador_beto"}'), usar(B))

    def test_invalid_types_are_errors(self):
        for value in (None, 3, [], "sin JSON", '{"accion":}', '{"accion":"otra"}'):
            with self.subTest(value=value):
                self.assertEqual(A._parsear_decision(value)["accion"], "error")

    def test_tool_name_must_be_string(self):
        for name in ('[]', '{}', 'null', '"inexistente"'):
            self.assertEqual(A._parsear_decision('{"accion":"usar_herramienta","herramienta":' + name + '}')["accion"], "error")

    def test_final_response_must_be_nonempty_text(self):
        for value in ('[]', '{}', 'null', '3', '" "'):
            self.assertEqual(A._parsear_decision('{"accion":"responder_final","respuesta":' + value + '}')["accion"], "error")

    def test_braces_inside_text_are_not_truncated(self):
        d = A._parsear_decision('```json\n{"accion":"responder_final","respuesta":"dato {x}"}\n```')
        self.assertEqual(d["respuesta"], "dato {x}")


class AgentTests(unittest.TestCase):
    def test_normal_exit_uses_structured_decision_not_planner_verdict(self):
        r, _ = ejecutar([usar(B), cerrar()])
        self.assertEqual(r["decision"]["etiqueta"], "urgente")
        self.assertTrue(r["respuesta"].startswith("URGENCIA: urgente"))
        self.assertEqual(r["texto_cierre_propuesto"], "URGENCIA: no_urgente")

    def test_all_exit_routes_agree(self):
        for steps in ([usar(B), usar(R), cerrar()], [usar(B), usar(R), usar(B)],
                      [usar(B)] + [{"accion": "error", "detalle": "prueba"}] * 4):
            r, _ = ejecutar(steps)
            self.assertEqual(r["decision"]["etiqueta"], "urgente")
            self.assertTrue(r["respuesta"].startswith("URGENCIA: urgente"))

    def test_valid_nonurgent_retained(self):
        r, _ = ejecutar([usar(B), usar(R), cerrar("texto distinto")], beto={"etiqueta": "no_urgente"})
        self.assertEqual(r["decision"]["etiqueta"], "no_urgente")
        self.assertIn("URGENCIA: no_urgente", r["respuesta"])

    def test_both_failures_never_become_routine(self):
        for steps in ([usar(B), usar(R), usar(B)], [usar(B), usar(R), cerrar()]):
            r, _ = ejecutar(steps, fallas=(B, R))
            self.assertIsNone(r["decision"]["etiqueta"])
            self.assertFalse(r["senal_disponible"])
            self.assertTrue(r["respuesta"].startswith("URGENCIA: no_determinable"))
            self.assertEqual(set(r["decision"]["tools_con_error"]), {B, R})

    def test_no_tools_no_invented_prediction(self):
        r, _ = ejecutar([cerrar("Respuesta fuera de alcance")])
        self.assertIsNone(r["decision"]["etiqueta"])
        self.assertEqual(r["decision"]["accion"], "revision_manual")

    def test_exhausted_invalid_planner_no_prediction(self):
        r, _ = ejecutar([{"accion": "error", "detalle": "prueba"}] * A.MAX_PASOS)
        self.assertEqual(r["decision"]["estado"], "sin_senal")

    def test_partial_failure_preserves_valid_signal_and_error(self):
        r, _ = ejecutar([usar(B), usar(R), cerrar()], fallas=(R,))
        self.assertEqual(r["decision"]["etiqueta"], "urgente")
        self.assertEqual(r["decision"]["estado"], "senal_parcial")
        self.assertIn("error", r["detalle_tools"][R])

    def test_repeated_tools_not_reexecuted(self):
        r, callbacks = ejecutar([usar(B), usar(B), usar(R), usar(R)])
        self.assertEqual([callbacks[B].call_count, callbacks[R].call_count], [1, 1])
        self.assertTrue(r["decision"]["desacuerdo"])

    def test_valve_preserves_raw_state_and_operational_escalation(self):
        r, _ = ejecutar([usar(B), usar(R), cerrar()], rag={"etiqueta": "urgente", "etiqueta_rag": "no_determinable",
                                                        "valvula": True, "respuesta": "URGENCIA: no_determinable"})
        self.assertEqual(r["decision"]["etiqueta"], "urgente")
        self.assertEqual(r["decision"]["etiqueta_rag_original"], "no_determinable")
        self.assertTrue(r["decision"]["escalamiento_por_valvula"])

    def test_tool_wrapper_retains_original_rag_label(self):
        raw = {"etiqueta_predicha": "urgente", "etiqueta_rag": "no_aplica", "valvula": False}
        with patch.object(A, "_obtener_rag", return_value=Mock(return_value=raw)):
            self.assertEqual(A.tool_buscar_informacion_dermatologica("q")["etiqueta_rag"], "no_aplica")

    def test_errored_label_is_not_valid_signal(self):
        d = dec.decidir({B: {"error": "failed", "etiqueta": "urgente"}})
        self.assertIsNone(d["etiqueta"])

    def test_context_sources_and_trace_preserved(self):
        r, _ = ejecutar([usar(B), usar(R), cerrar()])
        self.assertEqual(r["contexto"], ["dato de prueba"])
        self.assertEqual(r["fuentes"], ["f1"])
        self.assertEqual(len(r["pasos_debug"]), 3)
        self.assertEqual(r["tools_usadas"], [B, R])

    def test_invalid_tool_result_becomes_error_record(self):
        with patch.object(A, "HERRAMIENTAS", {B: {"funcion": lambda _: []}}), patch.object(
                A, "_preguntar_al_cerebro", side_effect=[(usar(B), ""), (usar(B), "")]):
            r = A.agente("q")
        self.assertFalse(r["senal_disponible"])
        self.assertIn("error", r["detalle_tools"][B])


if __name__ == "__main__":
    unittest.main()
