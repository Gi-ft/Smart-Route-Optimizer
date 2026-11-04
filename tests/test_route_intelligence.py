import sys
import unittest
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from route_intelligence import (  # noqa: E402
    assistant_answer,
    append_chat_message,
    bottleneck_analysis,
    build_context,
    compare_routes,
    evaluate_scenario,
    llm_answer,
    llm_configuration,
    parse_scenario_question,
    python_answer,
)


LOCATIONS = [
    {"location_id": "W", "role": "Warehouse", "location_name": "Warehouse", "latitude": -1.2, "longitude": 36.9},
    {"location_id": "A", "role": "Customer", "location_name": "Customer A", "latitude": -1.21, "longitude": 36.91},
]


def route(route_ids, distance, eta, legs):
    return {"route": route_ids, "route_string": " → ".join(route_ids), "total_distance_km": distance,
            "total_eta_min": eta, "leg_details": legs}


def sample_result():
    legs_short = [
        {"origin": "W", "destination": "A", "distance_km": 4.0, "predicted_eta_min": 10, "departure_hour": 8, "day_of_week": 0},
        {"origin": "A", "destination": "W", "distance_km": 5.0, "predicted_eta_min": "-", "departure_hour": 8, "day_of_week": 0},
    ]
    legs_fast = [
        {"origin": "W", "destination": "A", "distance_km": 4.2, "predicted_eta_min": 8, "departure_hour": 8, "day_of_week": 0},
        {"origin": "A", "destination": "W", "distance_km": 5.1, "predicted_eta_min": 9, "departure_hour": 8, "day_of_week": 0},
    ]
    short = route(["W", "A", "W"], 9, 20, legs_short)
    fast = route(["W", "A", "W"], 9.3, 17, legs_fast)
    return {"primary_key": "distance", "primary_result": short, "distance_result": short,
            "time_result": fast, "vehicle_type": "Motorcycle", "departure_day": "Monday",
            "departure_time": "08:00", "optimization_mode": "Shortest Distance",
            "scenario": "Typical simulated conditions"}


class RouteIntelligenceTests(unittest.TestCase):
  def test_openrouter_configuration_uses_secrets_and_environment(self):
    from route_intelligence import OPENROUTER_API_BASE

    config = llm_configuration(
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "provider/model-x"},
        environ={},
    )
    self.assertEqual(config["api_base"], "https://openrouter.ai/api/v1")
    self.assertEqual(config["api_base"], OPENROUTER_API_BASE)
    self.assertEqual(config["model"], "provider/model-x")
    self.assertTrue(config["available"])
    env_config = llm_configuration({}, {"OPENROUTER_API_KEY": "env-key", "OPENROUTER_MODEL": "openai/gpt-test"})
    self.assertEqual(env_config["model"], "openai/gpt-test")
    self.assertEqual(llm_configuration({}, {})["model"], "google/gemma-4-31b-it")

  def test_openrouter_success_uses_configured_model_and_timeout(self):
    import route_intelligence

    response = io.BytesIO(b'{"choices":[{"message":{"content":"Grounded explanation"}}]}')
    configuration = llm_configuration(
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "provider/model-x"}, {}
    )
    with patch.object(route_intelligence, "urlopen", return_value=response) as mocked:
        answer = llm_answer("Why this route?", {"ready": True}, "approved facts", configuration, timeout=7)
    self.assertEqual(answer, "Grounded explanation")
    request, = mocked.call_args.args
    self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
    self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
    self.assertEqual(json.loads(request.data)["model"], "provider/model-x")
    self.assertEqual(mocked.call_args.kwargs["timeout"], 7)

  def test_openrouter_provider_errors_and_timeouts_return_python_fallback(self):
    import route_intelligence

    context = build_context(sample_result(), LOCATIONS)
    configuration = {"api_key": "test-key", "api_base": "https://openrouter.ai/api/v1", "model": "provider/model-x"}
    failure_cases = [
        HTTPError("https://openrouter.ai/api/v1/chat/completions", 429, "rate limited", None, io.BytesIO(b'{"error":"rate limited"}')),
        TimeoutError("private timeout details"),
    ]
    for failure in failure_cases:
        with patch.object(route_intelligence, "urlopen", side_effect=failure):
            answer = assistant_answer("Compare routes", context, "LLM explanations", configuration)
        self.assertIn("Shortest-distance route", answer)
        self.assertNotIn("private timeout details", answer)
    provider_error = io.BytesIO(b'{"error":{"message":"private provider detail"}}')
    with patch.object(route_intelligence, "urlopen", return_value=provider_error):
        answer = assistant_answer("Compare routes", context, "LLM explanations", configuration)
    self.assertIn("Shortest-distance route", answer)
    self.assertNotIn("private provider detail", answer)

  def test_missing_openrouter_key_skips_request_and_uses_python(self):
    import route_intelligence

    context = build_context(sample_result(), LOCATIONS)
    configuration = llm_configuration({}, {})
    with patch.object(route_intelligence, "urlopen") as mocked:
        answer = assistant_answer("Compare routes", context, "LLM explanations", configuration)
    mocked.assert_not_called()
    self.assertIn("Shortest-distance route", answer)

  def test_context_with_and_without_optimization(self):
    absent = build_context(None, LOCATIONS)
    self.assertFalse(absent["ready"])
    self.assertIn("Optimize", absent["message"])
    context = build_context(sample_result(), LOCATIONS)
    self.assertTrue(context["ready"])
    self.assertEqual(context["shortest_distance"]["route"], ["W", "A", "W"])
    self.assertTrue(context["shortest_distance"]["legs"][0]["predicted_arrival"])


  def test_route_comparison_and_bottlenecks_handle_missing_eta(self):
    context = build_context(sample_result(), LOCATIONS)
    comparison = compare_routes(context)
    self.assertAlmostEqual(comparison["distance_difference_km"], 0.3)
    self.assertEqual(comparison["time_difference_min"], -3)
    findings = bottleneck_analysis(context)
    self.assertEqual(findings["longest_distance_leg"]["distance_km"], 5.1)
    self.assertEqual(findings["slowest_eta_leg"]["predicted_eta_min"], 10)
    self.assertEqual(len(findings["missing_eta_legs"]), 1)
    answer = python_answer("Which leg is slowest?", context)
    self.assertIn("unavailable", answer)


  def test_python_only_answer_requires_no_credentials(self):
    context = build_context(sample_result(), LOCATIONS)
    self.assertFalse(llm_configuration(environ={})["available"])
    answer = python_answer("Compare the two optimization methods", context)
    self.assertIn("predicted", answer)


  def test_scenario_validation_and_parameter_parsing(self):
    context = build_context(sample_result(), LOCATIONS)
    params = parse_scenario_question("What if I use a van and depart Saturday at 8 AM?", context)
    self.assertEqual(params["vehicle"], "Van")
    self.assertEqual(params["departure_hour"], 8)
    self.assertEqual(params["departure_day"], "Saturday")
    with self.assertRaisesRegex(ValueError, "Vehicle"):
        parse_scenario_question("What if I use a bicycle?", context)
    with self.assertRaisesRegex(ValueError, "12-hour"):
        parse_scenario_question("What if I depart at 13 PM?", context)


  def test_scenario_reuses_algorithms_returns_both_and_is_isolated(self):
    baseline = sample_result()
    original = repr(baseline)
    calls = []

    def distance_optimizer():
        calls.append("distance")
        return pd.DataFrame([{"route": ["W", "A", "W"], "route_string": "W -> A -> W", "total_distance_km": 9.0}])

    def eta_optimizer(vehicle, hour, day, scenario):
        calls.append((vehicle, hour, day, scenario))
        return [{"route": ["W", "A", "W"], "route_string": "W -> A -> W", "total_distance_km": 9.2,
                 "total_eta_min": 18.0, "leg_details": []}]

    def evaluate_distance(**kwargs):
        calls.append(kwargs)
        return 9.0, 19.0, []

    scenario = evaluate_scenario(
        {"vehicle": "van", "departure_hour": 8, "departure_day": "Saturday", "objective": "time"},
        baseline, distance_optimizer, eta_optimizer, evaluate_distance,
    )
    self.assertTrue(set(("distance_result", "time_result")) <= set(scenario))
    self.assertEqual(scenario["parameters"]["departure_day"], "Saturday")
    self.assertEqual(calls[0], "distance")
    self.assertEqual(calls[2][0:3], ("van", 8, 5))
    self.assertEqual(repr(baseline), original)
    with self.assertRaisesRegex(ValueError, "Vehicle"):
        evaluate_scenario({"vehicle": "bike", "departure_hour": 8, "departure_day": "Saturday"}, baseline, distance_optimizer, eta_optimizer, evaluate_distance)
    with self.assertRaisesRegex(ValueError, "hour"):
        evaluate_scenario({"vehicle": "van", "departure_hour": 24, "departure_day": "Saturday"}, baseline, distance_optimizer, eta_optimizer, evaluate_distance)


  def test_llm_provider_failure_falls_back_without_exposing_error(self):
    import route_intelligence

    def unavailable(*args, **kwargs):
        raise OSError("secret endpoint detail")

    original_urlopen = route_intelligence.urlopen
    route_intelligence.urlopen = unavailable
    configuration = {"api_key": "not-a-real-key", "api_base": "https://example.invalid/v1", "model": "test"}
    context = build_context(sample_result(), LOCATIONS)
    try:
        fallback = assistant_answer("Compare routes", context, "LLM explanations", configuration)
    finally:
        route_intelligence.urlopen = original_urlopen
    self.assertIn("Shortest-distance route", fallback)
    self.assertNotIn("secret endpoint detail", fallback)


  def test_missing_eta_is_retained_as_unavailable(self):
    context = build_context(sample_result(), LOCATIONS)
    self.assertIsNone(context["shortest_distance"]["legs"][1]["predicted_eta_min"])
    self.assertIn("unavailable", python_answer("Which leg is slowest?", context))


  def test_chat_state_updates_do_not_clear_optimization_results(self):
    state = {"optimization_result": sample_result(), "route_assistant_messages": []}
    original = repr(state["optimization_result"])
    append_chat_message(state, "user", "Compare the routes")
    append_chat_message(state, "assistant", "Python analysis response")
    self.assertEqual(len(state["route_assistant_messages"]), 2)
    self.assertEqual(repr(state["optimization_result"]), original)
