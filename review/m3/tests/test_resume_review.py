"""Pruebas offline de caché y reanudación; no ejecutan un juez real."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluacion_m3_reanudable as ev


def casos():
    return [{"id": "g1", "input": "q1", "tipo": "gold", "esperado": "urgente", "respuesta_referencia": "ref"},
            {"id": "g2", "input": "q2", "tipo": "gold", "esperado": "no_urgente", "respuesta_referencia": "ref"}]


def salida(q):
    return {"respuesta": "dato de prueba " + q, "contexto": ["contexto original"], "fuentes": ["d1"],
            "decision": {"etiqueta": "urgente" if q == "q1" else "no_urgente"}}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.fn = Mock(side_effect=salida)

    def init(self):
        return ev.congelar(self.d, casos(), self.fn, {"codigo": "v1"})

    def test_inference_resume_never_repeats_saved_outputs(self):
        a = self.init()
        b = self.init()
        self.assertEqual(a, b)
        self.assertEqual(self.fn.call_count, 2)

    def test_interrupted_inference_keeps_completed_cases(self):
        fn = Mock(side_effect=[salida("q1"), RuntimeError("controlado")])
        with self.assertRaises(RuntimeError):
            ev.congelar(self.d, casos(), fn, {})
        self.assertEqual(list(ev.leer(self.d / "inferencia.json")["salidas"]), ["g1"])
        fn = Mock(side_effect=salida)
        ev.congelar(self.d, casos(), fn, {})
        fn.assert_called_once_with("q2")

    def test_changed_input_or_system_refuses_resume(self):
        self.init()
        edited = casos()
        edited[0]["input"] = "otro"
        for c, firma in ((edited, {"codigo": "v1"}), (casos(), {"codigo": "v2"})):
            with self.assertRaises(ValueError):
                ev.congelar(self.d, c, self.fn, firma)
        self.assertEqual(self.fn.call_count, 2)

    def test_tampered_context_rejected_before_judge(self):
        self.init()
        s = ev.leer(self.d / "inferencia.json")
        s["salidas"]["g1"]["contexto"] = ["otro"]
        ev.guardar(self.d / "inferencia.json", s)
        judge = Mock(return_value=.5)
        with self.assertRaises(ValueError):
            ev.medir(self.d, judge, {})
        judge.assert_not_called()

    def test_duplicate_ids_rejected(self):
        data = casos()
        data[1]["id"] = "g1"
        with self.assertRaises(ValueError):
            ev.congelar(self.d, data, self.fn, {})
        self.fn.assert_not_called()

    def test_no_context_counter_substitute(self):
        r = salida("q1")
        r["contexto"] = 3
        with self.assertRaises(ValueError):
            ev.validar_salida(r)

    def test_no_signal_not_reported_as_nonurgent(self):
        s = self.init()
        s["salidas"]["g1"]["decision"]["etiqueta"] = None
        r = ev.resumen_harness(s)
        self.assertFalse(r["metricas_reportables"])
        self.assertIsNone(r["recall_urgente"])
        self.assertEqual(r["ids_sin_senal"], ["g1"])
        self.assertEqual(r["n_binarios"], 2)

    def test_harness_exact_match_and_recall(self):
        r = ev.resumen_harness(self.init())
        self.assertEqual((r["accuracy"], r["recall_urgente"], r["especificidad"]), (1., 1., 1.))

    def test_metric_resume_skips_valid_zero(self):
        self.init()
        judge = Mock(return_value=0.)
        r = ev.medir(self.d, judge, {"juez": "x"}, 1)
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(r["ragas"]["faithfulness"]["media"], 0.)
        r = ev.medir(self.d, judge, {"juez": "x"}, 100)
        self.assertEqual(judge.call_count, 8)
        self.assertEqual(r["jobs_esta_invocacion"], 7)
        ev.medir(self.d, judge, {"juez": "x"}, 100)
        self.assertEqual(judge.call_count, 8)
        self.assertEqual(self.fn.call_count, 2)

    def test_failure_persisted_and_stops_batch_then_only_pending(self):
        self.init()
        judge = Mock(side_effect=[.1, RuntimeError("cuota")])
        r = ev.medir(self.d, judge, {}, 100)
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(r["ragas"]["faithfulness"]["n_validos"], 1)
        second = Mock(return_value=.5)
        ev.medir(self.d, second, {}, 100)
        self.assertEqual(second.call_count, 7)
        self.assertNotIn("cuota", (self.d / "metricas.json").read_text())

    def test_nan_stays_unmeasured_not_zero(self):
        self.init()
        r = ev.medir(self.d, Mock(return_value=float("nan")), {}, 8)
        self.assertIsNone(r["ragas"]["faithfulness"]["media"])
        self.assertEqual(r["jobs_esta_invocacion"], 1)
        self.assertNotIn("NaN", (self.d / "metricas.json").read_text())

    def test_judge_fingerprint_change_rejected(self):
        self.init()
        ev.medir(self.d, Mock(return_value=.5), {"juez": "x"}, 1)
        fn = Mock()
        with self.assertRaises(ValueError):
            ev.medir(self.d, fn, {"juez": "y"}, 1)
        fn.assert_not_called()

    def test_mutated_snapshot_with_updated_hash_still_refuses_metric_merge(self):
        self.init()
        ev.medir(self.d, Mock(return_value=.5), {}, 1)
        s = ev.leer(self.d / "inferencia.json")
        s["salidas"]["g1"]["respuesta"] = "respuesta nueva"
        s["hashes"]["g1"] = ev.huella(s["salidas"]["g1"])
        ev.guardar(self.d / "inferencia.json", s)
        with self.assertRaises(ValueError):
            ev.medir(self.d, Mock(), {}, 8)

    def test_incomplete_snapshot_not_silently_scored(self):
        s = self.init()
        del s["salidas"]["g1"]
        del s["hashes"]["g1"]
        ev.guardar(self.d / "inferencia.json", s)
        with self.assertRaises(ValueError):
            ev.medir(self.d, Mock(), {}, 8)

    def test_lock_prevents_simultaneous_updates(self):
        with ev.bloquear(self.d):
            with self.assertRaises(FileExistsError):
                with ev.bloquear(self.d):
                    pass
        self.assertFalse((self.d / ".revision.lock").exists())

    def test_atomic_write_failure_keeps_previous_json(self):
        p = self.d / "data.json"
        ev.guardar(p, {"n": 1})
        with self.assertRaises(ValueError):
            ev.guardar(p, {"n": float("nan")})
        self.assertEqual(ev.leer(p), {"n": 1})

    def test_round_robin_not_all_urgent_prefix(self):
        data = [{"id": str(i), "tipo": "gold", "esperado": "urgente"} for i in range(5)]
        data += [{"id": "n", "tipo": "gold", "esperado": "no_urgente"},
                 {"id": "a", "tipo": "adversarial", "esperado": "no_aplica"}]
        ordered = ev.orden_intercalado(data)
        self.assertEqual(len({c["esperado"] for c in ordered[:3]}), 3)
        self.assertEqual(len(ordered), len(data))

    def test_zero_budget_does_not_call_judge(self):
        self.init()
        fn = Mock()
        r = ev.medir(self.d, fn, {}, 0)
        fn.assert_not_called()
        self.assertEqual(r["ragas"]["faithfulness"]["n_validos"], 0)

    def test_invalid_budgets(self):
        for n in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                ev.medir(self.d, Mock(), {}, n)

    def test_backend_passes_one_metric_frozen_response_and_original_context(self):
        import pandas as pd
        mods = {}
        def mod(name, **attrs):
            obj = types.ModuleType(name)
            obj.__dict__.update(attrs)
            mods[name] = obj
            return obj
        orig = mod("evaluar_m3", _parche_vertexai=Mock(), _crear_llm_juez=Mock(return_value="llm"),
                   _crear_embeddings_local=Mock(return_value="emb"))
        metrics = types.SimpleNamespace(**{m: types.SimpleNamespace(name=m) for m in ev.METRICAS})
        eval_fn = Mock(return_value=types.SimpleNamespace(to_pandas=lambda: pd.DataFrame({"faithfulness": [.4]})))
        mod("ragas", metrics=metrics, evaluate=eval_fn)
        mod("ragas.metrics", **vars(metrics))
        mod("ragas.llms", LangchainLLMWrapper=lambda x: x)
        mod("ragas.embeddings", LangchainEmbeddingsWrapper=lambda x: x)
        mod("ragas.run_config", RunConfig=lambda **kw: kw)
        mod("datasets", Dataset=types.SimpleNamespace(from_dict=lambda x: x))
        args = types.SimpleNamespace(proveedor="groq", juez="j", embeddings="e", timeout=20)
        with patch.dict(sys.modules, mods), patch.object(ev.metadata, "version", return_value="0.2.15"):
            scorer = ev.crear_evaluador(args)
            orig._crear_llm_juez.assert_not_called()
            self.assertEqual(scorer(casos()[0], salida("q1"), "faithfulness"), .4)
        call = eval_fn.call_args
        self.assertEqual(call.args[0]["retrieved_contexts"], [["contexto original"]])
        self.assertEqual(call.args[0]["response"], ["dato de prueba q1"])
        self.assertEqual(len(call.kwargs["metrics"]), 1)
        self.assertTrue(call.kwargs["raise_exceptions"])
        self.assertEqual(call.kwargs["run_config"]["max_workers"], 1)


if __name__ == "__main__":
    unittest.main()
