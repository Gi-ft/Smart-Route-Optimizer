"""Grounded Python analysis and optional language-model explanations.

This module contains no Streamlit dependencies so context and scenario logic can
be exercised independently of the dashboard.
"""

from __future__ import annotations

import json
import math
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
VEHICLES = {"motorcycle", "car", "van"}
UNAVAILABLE = {"", "-", "n/a", "na", "none", "null", "nan", "unavailable"}
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def numeric(value):
    """Return a finite float, or None for missing/placeholder values."""
    if value is None or str(value).strip().lower() in UNAVAILABLE:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rows(locations):
    if locations is None:
        return []
    if hasattr(locations, "to_dict"):
        return locations.to_dict(orient="records")
    return list(locations)


def _route_record(record, locations_by_id, start_hour=0, start_day=0):
    if not isinstance(record, dict):
        return None
    raw_legs = record.get("leg_details") or []
    legs = []
    elapsed = 0.0
    for item in raw_legs:
        distance = numeric(item.get("distance_km"))
        eta = numeric(item.get("predicted_eta_min"))
        hour_offset = int(elapsed // 60)
        minute = int(elapsed % 60)
        depart_hour = (start_hour + hour_offset) % 24
        depart_day = (start_day + int((start_hour * 60 + elapsed) // 1440)) % 7
        arrival_elapsed = elapsed + (eta or 0.0)
        arrival_hour = int((start_hour * 60 + arrival_elapsed) % 1440 // 60)
        arrival_minute = int((start_hour * 60 + arrival_elapsed) % 60)
        legs.append({
            "origin": item.get("origin"),
            "origin_name": locations_by_id.get(item.get("origin"), item.get("origin")),
            "destination": item.get("destination"),
            "destination_name": locations_by_id.get(item.get("destination"), item.get("destination")),
            "distance_km": distance,
            "predicted_eta_min": eta,
            "departure_hour": numeric(item.get("departure_hour")) if item.get("departure_hour") is not None else depart_hour,
            "departure_day": DAY_NAMES[int(numeric(item.get("day_of_week"))) % 7] if numeric(item.get("day_of_week")) is not None else DAY_NAMES[depart_day],
            "predicted_arrival": f"{DAY_NAMES[(start_day + int((start_hour * 60 + arrival_elapsed) // 1440)) % 7]} {arrival_hour:02d}:{arrival_minute:02d}" if eta is not None else None,
        })
        if eta is not None:
            elapsed += eta
    route = record.get("route")
    return {
        "route": list(route) if route else [],
        "route_string": record.get("route_string") or " → ".join(route or []),
        "total_distance_km": numeric(record.get("total_distance_km", record.get("total_distance"))),
        "total_eta_min": numeric(record.get("total_eta_min", record.get("total_eta"))),
        "legs": legs,
        "missing_eta_legs": [leg for leg in legs if leg["predicted_eta_min"] is None],
    }


def build_context(optimization_result, locations, vehicle=None, departure_day=None,
                  departure_time=None, objective=None, scenario=None):
    """Create a serializable assistant context from current app state."""
    location_rows = _rows(locations)
    location_by_id = {row.get("location_id"): row.get("location_name", row.get("location_id")) for row in location_rows}
    location_summary = [
        {"id": row.get("location_id"), "role": row.get("role"), "name": row.get("location_name"),
         "latitude": numeric(row.get("latitude")), "longitude": numeric(row.get("longitude"))}
        for row in location_rows
    ]
    if not optimization_result:
        return {
            "ready": False,
            "message": "Run Optimize delivery route first so I can answer from current route results.",
            "locations": location_summary,
            "limitations": ["ETA values are model predictions based on simulated training data, not live traffic."],
            "warnings": ["No optimization results are available in this session."],
        }

    time_parts = str(departure_time or optimization_result.get("departure_time", "00:00")).split(":")
    try:
        start_hour = int(time_parts[0])
    except (ValueError, IndexError):
        start_hour = 0
    day_name = departure_day or optimization_result.get("departure_day", "Monday")
    day_index = DAY_NAMES.index(day_name) if day_name in DAY_NAMES else 0
    distance = _route_record(optimization_result.get("distance_result"), location_by_id, start_hour, day_index)
    fastest = _route_record(optimization_result.get("time_result"), location_by_id, start_hour, day_index)
    primary = _route_record(optimization_result.get("primary_result") or optimization_result, location_by_id, start_hour, day_index)
    warnings = []
    if len(location_rows) != 7:
        warnings.append(f"Expected seven mapped locations; found {len(location_rows)}.")
    invalid_location_ids = [
        row["id"] for row in location_summary
        if row["latitude"] is None or row["longitude"] is None
    ]
    if invalid_location_ids:
        warnings.append("Missing or invalid coordinates for location(s): " + ", ".join(invalid_location_ids) + ".")
    for name, route in (("shortest-distance", distance), ("fastest-time", fastest)):
        if route and route["missing_eta_legs"]:
            warnings.append(f"{name} route has {len(route['missing_eta_legs'])} unavailable ETA value(s).")
    return {
        "ready": bool(primary),
        "vehicle": vehicle or optimization_result.get("vehicle_type"),
        "departure_day": day_name,
        "departure_time": departure_time or optimization_result.get("departure_time"),
        "departure_hour": start_hour,
        "objective": objective or optimization_result.get("optimization_mode"),
        "primary_key": optimization_result.get("primary_key", "distance"),
        "traffic_scenario": scenario or optimization_result.get("scenario"),
        "locations": location_summary,
        "primary": primary,
        "shortest_distance": distance,
        "fastest_time": fastest,
        "limitations": [
            "Road distances come from the mapped OpenStreetMap road network.",
            "Delivery durations are ETA model predictions trained on simulated data; there is no live traffic feed.",
            "Vehicle capacity, service time, time windows, and live road conditions are not modeled.",
        ],
        "warnings": warnings,
    }


def compare_routes(context):
    """Return numerical differences with unavailable values left as None."""
    short = context.get("shortest_distance") or {}
    fast = context.get("fastest_time") or {}
    sd, fd = numeric(short.get("total_distance_km")), numeric(fast.get("total_distance_km"))
    st, ft = numeric(short.get("total_eta_min")), numeric(fast.get("total_eta_min"))
    return {
        "distance_difference_km": fd - sd if fd is not None and sd is not None else None,
        "time_difference_min": ft - st if ft is not None and st is not None else None,
        "shortest_distance": {"distance_km": sd, "eta_min": st, "route": short.get("route_string")},
        "fastest_time": {"distance_km": fd, "eta_min": ft, "route": fast.get("route_string")},
    }


def bottleneck_analysis(context):
    routes = [context.get("primary"), context.get("shortest_distance"), context.get("fastest_time")]
    all_legs = []
    seen = set()
    for route in routes:
        if not route:
            continue
        for leg in route.get("legs", []):
            key = (tuple(route.get("route", [])), leg.get("origin"), leg.get("destination"),
                   leg.get("distance_km"), leg.get("predicted_eta_min"))
            if key not in seen:
                seen.add(key)
                all_legs.append(leg)
    distance_legs = [leg for leg in all_legs if leg["distance_km"] is not None]
    eta_legs = [leg for leg in all_legs if leg["predicted_eta_min"] is not None]
    return {
        "longest_distance_leg": max(distance_legs, key=lambda leg: leg["distance_km"], default=None),
        "slowest_eta_leg": max(eta_legs, key=lambda leg: leg["predicted_eta_min"], default=None),
        "largest_time_contribution_leg": max(eta_legs, key=lambda leg: leg["predicted_eta_min"], default=None),
        "missing_eta_legs": [leg for leg in all_legs if leg["predicted_eta_min"] is None],
    }


def python_answer(question, context):
    """Answer common operational questions deterministically from context."""
    if not context.get("ready"):
        return context.get("message", "Run route optimization first.")
    text = question.lower()
    comparison = compare_routes(context)
    primary = context.get("primary") or {}
    if any(word in text for word in ("longest", "slowest", "bottleneck", "takes the longest", "takes longest")):
        findings = bottleneck_analysis(context)
        lines = []
        for title, key, metric, unit in (
            ("Longest road-distance leg", "longest_distance_leg", "distance_km", "km"),
            ("Slowest predicted leg", "slowest_eta_leg", "predicted_eta_min", "min"),
        ):
            leg = findings[key]
            if leg:
                lines.append(f"{title}: {leg['origin_name']} → {leg['destination_name']} at {leg[metric]:.2f} {unit}.")
        if findings["missing_eta_legs"]:
            lines.append(f"{len(findings['missing_eta_legs'])} leg ETA value(s) are unavailable and excluded from time ranking.")
        return "\n\n".join(lines) or "No valid leg measurements are available."
    if any(word in text for word in ("missing", "unavailable", "unknown eta")):
        missing = bottleneck_analysis(context)["missing_eta_legs"]
        if not missing:
            return "All delivery legs in the current route results have a numeric predicted ETA. These remain model estimates, not live traffic measurements."
        legs = ", ".join(f"{leg['origin_name']} → {leg['destination_name']}" for leg in missing)
        return f"{len(missing)} delivery leg ETA value(s) are unavailable: {legs}. I excluded those values from the slowest-leg analysis."
    if any(word in text for word in ("compare", "comparison", "trade-off", "tradeoff", "both method")):
        if comparison["distance_difference_km"] is None or comparison["time_difference_min"] is None:
            return "One or more route totals are unavailable, so a complete comparison cannot be calculated."
        d, t = comparison["distance_difference_km"], comparison["time_difference_min"]
        return (f"Shortest-distance route: {comparison['shortest_distance']['distance_km']:.2f} km and "
                f"{comparison['shortest_distance']['eta_min']:.2f} predicted min. Fastest-time route: "
                f"{comparison['fastest_time']['distance_km']:.2f} km and {comparison['fastest_time']['eta_min']:.2f} predicted min. "
                f"The fastest-time route uses {d:+.2f} km and changes predicted duration by {t:+.2f} min versus shortest distance. "
                "Choose based on whether distance or predicted delivery time is the priority.")
    if "recommend" in text or "should i" in text or "what should" in text:
        if any(word in text for word in ("fast", "speed", "time")):
            selected = context.get("fastest_time") or {}
            label, reason = "fastest-time", "it has the lowest model-predicted total duration among the evaluated route orders"
        elif any(word in text for word in ("distance", "short", "fuel", "mileage")):
            selected = context.get("shortest_distance") or {}
            label, reason = "shortest-distance", "it has the lowest total road distance among the evaluated route orders"
        else:
            selected = primary
            label, reason = str(context.get("objective", "selected")).lower(), "it matches the objective selected in the dashboard"
        if selected.get("total_distance_km") is None or selected.get("total_eta_min") is None:
            return "I can't make a route recommendation because one or more route totals are unavailable."
        return (f"For this priority, I recommend the {label} route ({selected['route_string']}): "
                f"{selected['total_distance_km']:.2f} km and {selected['total_eta_min']:.2f} model-predicted min. "
                f"Evidence: {reason}. ETA values are predictions based on simulated training data.")
    if any(word in text for word in ("why", "selected")):
        key = "shortest_distance" if "short" in text or "distance" in text else "fastest_time" if "fast" in text or "time" in text else "shortest_distance" if context.get("primary_key") == "distance" else "fastest_time"
        route = context.get(key) if key else primary
        objective = "shortest-distance" if key == "shortest_distance" else "fastest-time" if key == "fastest_time" else str(context.get("objective", "selected"))
        if route:
            criterion = (
                "it minimizes road distance across the 720 customer orders"
                if objective == "shortest-distance"
                else "it minimizes cumulative model-predicted delivery duration across the 720 customer orders"
                if objective == "fastest-time"
                else "it was selected under the dashboard's configured objective"
            )
            return (f"The {objective} route was selected because {criterion}: {route.get('route_string', 'unavailable')}. Its road distance is "
                    f"{route['total_distance_km']:.2f} km and its model-predicted duration is "
                    f"{route['total_eta_min']:.2f} min." if route.get("total_distance_km") is not None and route.get("total_eta_min") is not None
                    else f"The {objective} route is {route.get('route_string', 'unavailable')}; one or more route totals are unavailable.")
    if any(word in text for word in ("limitation", "limit", "traffic", "accuracy", "reliable")):
        return "The route uses mapped road-network distances. ETAs are model predictions trained on simulated durations, not live traffic observations. Service time, capacity, delivery windows, and current road conditions are not represented."
    if any(word in text for word in ("vehicle", "motorcycle", "van", "car")):
        return "Vehicle scenarios are evaluated with the application's ETA model for each supported vehicle. The result is an estimate from simulated training data, not a measured vehicle-performance guarantee. Ask ‘What if I use a van?’ to run a scenario against the current settings."
    findings = bottleneck_analysis(context)
    return (f"Current {context.get('objective', 'route')} recommendation: {primary.get('route_string', 'unavailable')}, "
            f"{primary.get('total_distance_km') if primary.get('total_distance_km') is not None else 'unavailable'} km and "
            f"{primary.get('total_eta_min') if primary.get('total_eta_min') is not None else 'unavailable'} model-predicted minutes. "
            f"I found {len(findings['missing_eta_legs'])} unavailable leg ETA value(s). You can ask for a route comparison, a bottleneck, or a what-if scenario.")


def parse_scenario_question(question, baseline):
    """Extract and validate the small supported set of scenario parameters."""
    text = question.lower()
    vehicle = baseline.get("vehicle", "Motorcycle")
    for candidate in VEHICLES:
        if re.search(rf"\b{candidate}\b", text):
            vehicle = candidate.title()
            break
    if re.search(r"\b(bicycle|bike|truck|bus|scooter)\b", text):
        raise ValueError("Vehicle must be Motorcycle, Car, or Van.")
    hour = baseline.get("departure_hour", 0)
    time24_match = re.search(r"\b(?:at\s*)?(\d{1,2}):(\d{2})\b", text)
    time_match = re.search(r"\b(?:at\s*)?(\d{1,2})(?::\d{2})?\s*(am|pm)\b", text)
    if time_match:
        raw = int(time_match.group(1))
        if not 1 <= raw <= 12:
            raise ValueError("Use a valid 12-hour time such as 8 AM or 5 PM.")
        hour = raw % 12 + (12 if time_match.group(2) == "pm" else 0)
    elif time24_match:
        hour, minute = int(time24_match.group(1)), int(time24_match.group(2))
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("Departure time must be a valid 24-hour time.")
    else:
        hour_match = re.search(r"\b(?:hour|departure hour)\s*(?:to|=|of)?\s*(\d{1,2})\b", text)
        if hour_match:
            hour = int(hour_match.group(1))
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("Departure hour must be between 0 and 23.")
    day = baseline.get("departure_day", "Monday")
    for candidate in DAY_NAMES:
        if re.search(rf"\b{candidate.lower()}\b", text):
            day = candidate
            break
    if day not in DAY_NAMES:
        raise ValueError("Choose a weekday from Monday through Sunday.")
    if vehicle.lower() not in VEHICLES:
        raise ValueError("Vehicle must be Motorcycle, Car, or Van.")
    objective = baseline.get("primary_key", "distance")
    if "prioritize distance" in text or "prioritise distance" in text:
        objective = "distance"
    elif "prioritize time" in text or "prioritise time" in text or "fastest" in text or "delivery time" in text:
        objective = "time"
    elif "shortest" in text:
        objective = "distance"
    return {"vehicle": vehicle, "departure_hour": hour, "departure_day": day, "objective": objective}


def evaluate_scenario(parameters, baseline, optimize_distance, optimize_time, evaluate_distance):
    """Re-run existing optimizers and return both scenario routes and deltas."""
    vehicle = str(parameters.get("vehicle", "")).lower()
    hour = parameters.get("departure_hour")
    day = parameters.get("departure_day")
    objective = parameters.get("objective", "distance")
    if vehicle not in VEHICLES:
        raise ValueError("Vehicle must be Motorcycle, Car, or Van.")
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("Departure hour must be between 0 and 23.")
    if day not in DAY_NAMES:
        raise ValueError("Choose a weekday from Monday through Sunday.")
    if objective not in ("distance", "time"):
        raise ValueError("Objective must be distance or time.")
    day_index = DAY_NAMES.index(day)
    base_scenario = baseline.get("scenario", "Typical simulated conditions")
    distance_ranked = optimize_distance()
    distance_best = distance_ranked.iloc[0]
    distance_total, distance_eta, distance_legs = evaluate_distance(
        route=distance_best["route"], vehicle_type=vehicle, start_hour=hour,
        start_day=day_index, scenario=base_scenario,
    )
    fastest = optimize_time(vehicle, hour, day_index, base_scenario)[0]
    results = {
        "parameters": {"vehicle": vehicle.title(), "departure_hour": hour, "departure_day": day, "objective": objective, "scenario": base_scenario},
        "distance_result": {"route": list(distance_best["route"]), "route_string": distance_best["route_string"], "total_distance_km": numeric(distance_best["total_distance_km"]), "total_eta_min": numeric(distance_eta), "leg_details": distance_legs},
        "time_result": {"route": list(fastest["route"]), "route_string": fastest["route_string"], "total_distance_km": numeric(fastest["total_distance_km"]), "total_eta_min": numeric(fastest["total_eta_min"]), "leg_details": fastest["leg_details"]},
    }
    for key in ("distance_result", "time_result"):
        base = baseline.get(key) or {}
        current = results[key]
        current["delta_distance_km"] = current["total_distance_km"] - numeric(base.get("total_distance_km")) if numeric(base.get("total_distance_km")) is not None else None
        current["delta_eta_min"] = current["total_eta_min"] - numeric(base.get("total_eta_min")) if numeric(base.get("total_eta_min")) is not None else None
    return results


def llm_configuration(secrets=None, environ=None):
    """Read optional OpenRouter credentials/model without exposing secrets."""
    environ = os.environ if environ is None else environ
    secrets = secrets or {}
    def read_setting(name):
        try:
            value = secrets.get(name)
        except Exception:
            value = None
        return value or environ.get(name)

    api_key = read_setting("OPENROUTER_API_KEY")
    model = read_setting("OPENROUTER_MODEL") or "google/gemma-4-31b-it"
    return {
        "api_key": api_key,
        "api_base": OPENROUTER_API_BASE,
        "model": model,
        "available": bool(api_key),
    }


def llm_answer(question, context, python_analysis, configuration, timeout=15):
    """Ask OpenRouter to explain Python-approved facts; fail closed to Python."""
    if not configuration.get("api_key"):
        return None
    payload = {
        "model": configuration["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a route intelligence assistant. Explain only the provided structured application context and Python analysis. Do not calculate, infer missing values, invent traffic conditions, or claim simulated ETA is live traffic. If a value is null, say it is unavailable. Scenario outcomes must be supplied by Python."},
            {"role": "user", "content": json.dumps({"question": question, "context": context, "approved_python_analysis": python_analysis}, ensure_ascii=False)},
        ],
    }
    request = Request(
        configuration["api_base"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {configuration['api_key']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("error"):
            return None
        answer = data["choices"][0]["message"]["content"]
        return answer if isinstance(answer, str) and answer.strip() else None
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def assistant_answer(question, context, mode="Python analysis", configuration=None):
    """Return an LLM explanation when available, otherwise grounded Python text."""
    analysis = python_answer(question, context)
    if mode.startswith("LLM"):
        explanation = llm_answer(question, context, analysis, configuration or {})
        if explanation:
            return explanation
    return analysis


def append_chat_message(session_state, role, content):
    """Append assistant history without touching active optimization state."""
    messages = session_state.setdefault("route_assistant_messages", [])
    messages.append({"role": role, "content": content})
    return messages
