#!/usr/bin/env python3
"""Generate a synthetic SFT dataset for OpenUI-Lang component generation."""

import argparse
import datetime
import json
import random
from collections import Counter


SYSTEM_PROMPT = (
    "You generate OpenUI Lang from a user query and a structured tool result. "
    "Use only the values from the tool result. Do not invent data. Return only OpenUI Lang "
    "assignment statements, without explanations or markdown. Start with root = Root([...])."
)

DOMAINS = {
    "weather": [
        {"name": "Temperatur", "unit": "°C", "min": -5.0, "max": 34.0, "decimals": 1, "aggregation": "average", "threshold": 30.0},
        {"name": "Niederschlag", "unit": "mm", "min": 0, "max": 180, "decimals": 0, "aggregation": "sum", "threshold": 80},
        {"name": "Sonnenstunden", "unit": "h", "min": 40, "max": 2300, "decimals": 0, "aggregation": "sum", "threshold": 1800},
        {"name": "Windgeschwindigkeit", "unit": "km/h", "min": 3.0, "max": 72.0, "decimals": 1, "aggregation": "average", "threshold": 50.0},
        {"name": "Luftfeuchtigkeit", "unit": "%", "min": 28, "max": 96, "decimals": 0, "aggregation": "average", "threshold": 85},
    ],
    "air_quality": [
        {"name": "PM10", "unit": "µg/m³", "min": 8, "max": 90, "decimals": 0, "aggregation": "average", "threshold": 50},
        {"name": "PM2.5", "unit": "µg/m³", "min": 4, "max": 55, "decimals": 1, "aggregation": "average", "threshold": 25},
        {"name": "NO2", "unit": "µg/m³", "min": 8, "max": 85, "decimals": 0, "aggregation": "average", "threshold": 40},
        {"name": "Ozon", "unit": "µg/m³", "min": 20, "max": 190, "decimals": 0, "aggregation": "average", "threshold": 120},
        {"name": "Luftqualitätsindex", "unit": "AQI", "min": 15, "max": 150, "decimals": 0, "aggregation": "average", "threshold": 100},
    ],
    "water_quality": [
        {"name": "pH-Wert", "unit": "pH", "min": 6.4, "max": 8.8, "decimals": 1, "aggregation": "average", "threshold": 8.5},
        {"name": "Trübung", "unit": "NTU", "min": 0.2, "max": 18.0, "decimals": 1, "aggregation": "average", "threshold": 5.0},
        {"name": "Wassertemperatur", "unit": "°C", "min": 4.0, "max": 25.0, "decimals": 1, "aggregation": "average", "threshold": 22.0},
        {"name": "Gelöster Sauerstoff", "unit": "mg/l", "min": 5.0, "max": 13.0, "decimals": 1, "aggregation": "average", "threshold": 6.0},
        {"name": "Nitrat", "unit": "mg/l", "min": 1.0, "max": 42.0, "decimals": 1, "aggregation": "average", "threshold": 25.0},
    ],
    "traffic": [
        {"name": "Fahrzeuganzahl", "unit": "Fahrzeuge", "min": 1200, "max": 85000, "decimals": 0, "aggregation": "sum", "threshold": 60000},
        {"name": "Durchschnittsgeschwindigkeit", "unit": "km/h", "min": 12.0, "max": 64.0, "decimals": 1, "aggregation": "average", "threshold": 20.0},
        {"name": "Staudauer", "unit": "min", "min": 0, "max": 240, "decimals": 0, "aggregation": "sum", "threshold": 90},
        {"name": "Straßenauslastung", "unit": "%", "min": 12, "max": 98, "decimals": 0, "aggregation": "average", "threshold": 85},
        {"name": "ÖPNV-Verspätung", "unit": "min", "min": 0, "max": 35, "decimals": 0, "aggregation": "average", "threshold": 10},
    ],
    "public_administration": [
        {"name": "Anträge", "unit": "Anträge", "min": 80, "max": 12000, "decimals": 0, "aggregation": "sum", "threshold": 9000},
        {"name": "Bearbeitungszeit", "unit": "Tage", "min": 2, "max": 45, "decimals": 0, "aggregation": "average", "threshold": 21},
        {"name": "Terminauslastung", "unit": "%", "min": 35, "max": 100, "decimals": 0, "aggregation": "average", "threshold": 90},
        {"name": "Wartezeit", "unit": "min", "min": 3, "max": 95, "decimals": 0, "aggregation": "average", "threshold": 45},
        {"name": "Online-Anteil", "unit": "%", "min": 20, "max": 96, "decimals": 0, "aggregation": "average", "threshold": 75},
    ],
    "energy": [
        {"name": "Stromverbrauch", "unit": "MWh", "min": 150, "max": 9800, "decimals": 0, "aggregation": "sum", "threshold": 7500},
        {"name": "Solarerzeugung", "unit": "MWh", "min": 20, "max": 3600, "decimals": 0, "aggregation": "sum", "threshold": 2500},
        {"name": "Heizbedarf", "unit": "MWh", "min": 80, "max": 7200, "decimals": 0, "aggregation": "sum", "threshold": 5400},
        {"name": "CO2-Emissionen", "unit": "t", "min": 12, "max": 4200, "decimals": 0, "aggregation": "sum", "threshold": 3000},
        {"name": "Netzlast", "unit": "MW", "min": 35, "max": 850, "decimals": 0, "aggregation": "average", "threshold": 700},
    ],
    "population": [
        {"name": "Einwohner", "unit": "Personen", "min": 2500, "max": 3800000, "decimals": 0, "aggregation": "count", "threshold": 1500000},
        {"name": "Geburten", "unit": "Geburten", "min": 20, "max": 48000, "decimals": 0, "aggregation": "sum", "threshold": 30000},
        {"name": "Sterbefälle", "unit": "Sterbefälle", "min": 20, "max": 46000, "decimals": 0, "aggregation": "sum", "threshold": 25000},
        {"name": "Wanderungssaldo", "unit": "Personen", "min": -6000, "max": 26000, "decimals": 0, "aggregation": "sum", "threshold": 15000},
        {"name": "Durchschnittsalter", "unit": "Jahre", "min": 35.0, "max": 49.0, "decimals": 1, "aggregation": "average", "threshold": 45.0},
    ],
    "finance_budget": [
        {"name": "Budget", "unit": "Mio. €", "min": 8, "max": 9500, "decimals": 0, "aggregation": "sum", "threshold": 7000},
        {"name": "Ausgaben", "unit": "Mio. €", "min": 5, "max": 9200, "decimals": 0, "aggregation": "sum", "threshold": 6800},
        {"name": "Einnahmen", "unit": "Mio. €", "min": 5, "max": 9800, "decimals": 0, "aggregation": "sum", "threshold": 7000},
        {"name": "Defizit", "unit": "Mio. €", "min": 0, "max": 1400, "decimals": 0, "aggregation": "sum", "threshold": 600},
        {"name": "Investitionsvolumen", "unit": "Mio. €", "min": 2, "max": 2600, "decimals": 0, "aggregation": "sum", "threshold": 1500},
    ],
}

LOCATIONS = [
    "München",
    "Berlin",
    "Hamburg",
    "Köln",
    "Frankfurt am Main",
    "Schwabing-Freimann",
    "Sendling",
    "Moosach",
    "Bogenhausen",
    "Maxvorstadt",
    "Isar",
    "Rhein",
    "Elbe",
    "Donau",
    "Bürgerbüro Mitte",
    "Bürgerbüro Pasing",
    "Landsberger Straße",
    "Leopoldstraße",
    "Rosenheimer Straße",
]

DISTRICTS = [
    "Schwabing-Freimann",
    "Sendling",
    "Moosach",
    "Bogenhausen",
    "Maxvorstadt",
    "Neuhausen-Nymphenburg",
    "Au-Haidhausen",
    "Laim",
    "Giesing",
    "Pasing-Obermenzing",
]

OFFICES = [
    "Bürgerbüro Mitte",
    "Bürgerbüro Pasing",
    "Bürgerbüro Riesenfeldstraße",
    "Bürgerbüro Orleansplatz",
    "KVR Hauptstelle",
    "Sozialbürgerhaus Nord",
    "Referat für Stadtplanung",
    "Stadtkämmerei",
]

MONTHS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
DATA_SHAPES = [
    "scalar",
    "comparison",
    "time_series_daily",
    "time_series_monthly",
    "ranking",
    "threshold",
    "percentage",
    "table",
    "multi_kpi",
    "geo_values",
]
COMPONENT_BY_SHAPE = {
    "scalar": "MetricGrid",
    "comparison": "BarChart",
    "time_series_daily": "Histogram",
    "time_series_monthly": "BarChart",
    "ranking": "BarChart",
    "threshold": "Notice",
    "percentage": "MetricGrid",
    "table": "DataTable",
    "multi_kpi": "MetricGrid",
    "geo_values": "BarChart",
}


def js_string(value):
    return json.dumps(value, ensure_ascii=False)


def format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def js_prop_value(value):
    if isinstance(value, str):
        return js_string(value)
    if isinstance(value, list):
        return js_array(value)
    if isinstance(value, dict):
        return js_object(value)
    return format_number(value)


def js_object(obj):
    pairs = [f"{key}: {js_prop_value(value)}" for key, value in obj.items()]
    return "{ " + ", ".join(pairs) + " }"


def js_array(items, indent="    "):
    if not items:
        return "[]"
    lines = ["["]
    for index, item in enumerate(items):
        suffix = "," if index < len(items) - 1 else ""
        lines.append(f"{indent}{js_prop_value(item)}{suffix}")
    lines.append("  ]")
    return "\n".join(lines)


def value_for_metric(rng, metric):
    raw = rng.uniform(metric["min"], metric["max"])
    if metric["decimals"] == 0:
        return int(round(raw))
    return round(raw, metric["decimals"])


def nearby_value(rng, metric, base=None):
    if base is None:
        return value_for_metric(rng, metric)
    span = metric["max"] - metric["min"]
    delta = rng.uniform(-0.18 * span, 0.18 * span)
    value = max(metric["min"], min(metric["max"], base + delta))
    if metric["decimals"] == 0:
        return int(round(value))
    return round(value, metric["decimals"])


def choose_context(rng, domain=None):
    domain = domain or rng.choice(list(DOMAINS.keys()))
    metric = rng.choice(DOMAINS[domain])
    return domain, metric, rng.choice(LOCATIONS), rng.randint(2018, 2025)


def row(user_query, tool_result, assistant, domain, data_shape, component):
    content = user_query + "\n\nTool result:\n" + json.dumps(tool_result, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {"domain": domain, "data_shape": data_shape, "component": component},
    }


def openui_rows(rows):
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def metric_value(value, unit):
    return f"{format_number(value)} {unit}".strip()


def render_metric_grid(title, metrics, summary=None):
    children = [f"m{index + 1}" for index in range(len(metrics))]
    root_children = ["summary", "metrics"] if summary else ["metrics"]
    lines = [f"root = Root([{', '.join(root_children)}])"]
    if summary:
        lines.append(f"summary = InsightCard({js_string(title)}, {js_string(summary)})")
    lines.append(f"metrics = MetricGrid([{', '.join(children)}])")
    for ref, metric in zip(children, metrics):
        args = [
            js_string(metric["label"]),
            js_string(metric["value"]),
        ]
        if metric.get("caption"):
            args.append(js_string(metric["caption"]))
        lines.append(f"{ref} = Metric({', '.join(args)})")
    return "\n".join(lines)


def render_insight(title, body):
    return "\n".join(
        [
            "root = Root([summary])",
            f"summary = InsightCard({js_string(title)}, {js_string(body)})",
        ]
    )


def render_notice(title, message, tone="info", metrics=None):
    root_children = ["notice", "metrics"] if metrics else ["notice"]
    lines = [f"root = Root([{', '.join(root_children)}])"]
    lines.append(f"notice = Notice({js_string(f'{title}: {message}')}, {js_string(tone)})")
    if metrics:
        refs = [f"m{index + 1}" for index in range(len(metrics))]
        lines.append(f"metrics = MetricGrid([{', '.join(refs)}])")
        for ref, metric in zip(refs, metrics):
            lines.append(
                f"{ref} = Metric({js_string(metric['label'])}, {js_string(metric['value'])}, {js_string(metric.get('caption', ''))})"
            )
    return "\n".join(lines)


def render_data_table(title, rows):
    return "\n".join(
        [
            "root = Root([table])",
            f"table = DataTable({js_string(title)}, {openui_rows(rows)})",
        ]
    )


def render_chart(title, x_column, y_column, rows, summary=None):
    root_children = ["summary", "chart"] if summary else ["chart"]
    lines = [f"root = Root([{', '.join(root_children)}])"]
    if summary:
        lines.append(f"summary = InsightCard({js_string(title)}, {js_string(summary)})")
    lines.append(
        f"chart = BarChart({js_string(title)}, {js_string(x_column)}, {js_string(y_column)}, {openui_rows(rows)})"
    )
    return "\n".join(lines)


def render_histogram(title, column, values, summary=None):
    root_children = ["summary", "histogram"] if summary else ["histogram"]
    lines = [f"root = Root([{', '.join(root_children)}])"]
    if summary:
        lines.append(f"summary = InsightCard({js_string(title)}, {js_string(summary)})")
    lines.append(f"histogram = Histogram({js_string(title)}, {js_string(column)}, {openui_rows(values)})")
    return "\n".join(lines)


def query_from_templates(rng, templates, **kwargs):
    return rng.choice(templates).format(**kwargs)


def generate_scalar_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    value = value_for_metric(rng, metric)
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "aggregation": metric["aggregation"],
        "value": value,
        "unit": metric["unit"],
    }
    query = query_from_templates(
        rng,
        [
            "Wie hoch war {metric} in {location} im Jahr {year}?",
            "Wie viele {metric} gab es in {location} im Jahr {year}?",
            "Zeige mir den Wert für {metric} in {location} {year}.",
        ],
        metric=metric["name"],
        location=location,
        year=year,
    )
    title = f"{metric['name']} in {location} {year}"
    description = f"{'Gesamtwert' if metric['aggregation'] == 'sum' else 'Wert'} für das Jahr {year}"
    assistant = render_metric_grid(
        title,
        [{"label": metric["name"], "value": metric_value(value, metric["unit"]), "caption": description}],
        f"{metric['name']} wurde für {location} im Jahr {year} gemeldet.",
    )
    return row(query, tool_result, assistant, domain, "scalar", "MetricGrid")


def generate_comparison_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    previous_year = year - 1
    current = value_for_metric(rng, metric)
    previous = nearby_value(rng, metric, current)
    delta = round(current - previous, metric["decimals"])
    if metric["decimals"] == 0:
        delta = int(delta)
    direction = "up" if delta > 0 else "down" if delta < 0 else "neutral"
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "current": {"year": year, "value": current},
        "previous": {"year": previous_year, "value": previous},
        "unit": metric["unit"],
    }
    query = query_from_templates(
        rng,
        [
            "Vergleiche {metric} in {location} in {year} mit {previous_year}.",
            "Wie hat sich {metric} in {location} gegenüber {previous_year} verändert?",
            "Zeige den Jahresvergleich für {metric} in {location}.",
        ],
        metric=metric["name"],
        location=location,
        year=year,
        previous_year=previous_year,
    )
    title = f"{metric['name']} im Jahresvergleich in {location}"
    chart_rows = [
        {"year": str(previous_year), "value": previous},
        {"year": str(year), "value": current},
    ]
    summary = f"Der Unterschied beträgt {metric_value(delta, metric['unit'])}; Richtung: {direction}."
    assistant = render_chart(title, "year", "value", chart_rows, summary)
    return row(query, tool_result, assistant, domain, "comparison", "BarChart")


def generate_daily_series_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    days = rng.randint(7, 14)
    start = datetime.date(year, rng.randint(1, 12), rng.randint(1, 14))
    base = value_for_metric(rng, metric)
    values = []
    current = base
    for offset in range(days):
        current = nearby_value(rng, metric, current)
        values.append({"date": (start + datetime.timedelta(days=offset)).isoformat(), "value": current})
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "time_granularity": "daily",
        "unit": metric["unit"],
        "values": values,
    }
    query = query_from_templates(
        rng,
        [
            "Zeige mir die Entwicklung von {metric} in {location}.",
            "Wie hat sich {metric} in {location} in der letzten Woche entwickelt?",
            "Erstelle einen Tagesverlauf für {metric} in {location}.",
        ],
        metric=metric["name"],
        location=location,
    )
    title = f"{metric['name']}verlauf in {location}"
    assistant = render_histogram(
        title,
        "value",
        [item["value"] for item in values],
        f"Tageswerte für {metric['name']} in {location}; Einheit: {metric['unit']}.",
    )
    return row(query, tool_result, assistant, domain, "time_series_daily", "Histogram")


def generate_monthly_series_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    values = [{"month": month, "value": value_for_metric(rng, metric)} for month in MONTHS]
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "time_granularity": "monthly",
        "unit": metric["unit"],
        "values": values,
    }
    query = query_from_templates(
        rng,
        [
            "Zeige mir {metric} pro Monat für {year} in {location}.",
            "Wie verteilen sich die Monatswerte von {metric} in {location} {year}?",
            "Erstelle ein Monatsdiagramm für {metric} in {location}.",
        ],
        metric=metric["name"],
        location=location,
        year=year,
    )
    title = f"{metric['name']} pro Monat in {location} {year}"
    assistant = render_chart(title, "month", "value", values, f"Monatswerte in {metric['unit']}.")
    return row(query, tool_result, assistant, domain, "time_series_monthly", "BarChart")


def generate_ranking_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    labels = rng.sample(DISTRICTS, rng.randint(5, 10))
    values = [{"label": label, "value": value_for_metric(rng, metric)} for label in labels]
    values.sort(key=lambda item: item["value"], reverse=True)
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "unit": metric["unit"],
        "rank_by": "value_desc",
        "values": values,
    }
    query = query_from_templates(
        rng,
        [
            "Welche Stadtbezirke hatten die höchsten Werte bei {metric}?",
            "Zeige ein Ranking der Bezirke nach {metric} in {year}.",
            "Wo war {metric} in {location} am höchsten?",
        ],
        metric=metric["name"],
        location=location,
        year=year,
    )
    title = f"{metric['name']} nach Stadtbezirk {year}"
    assistant = render_chart(title, "label", "value", values, f"Ranking nach {metric['name']} in {metric['unit']}.")
    return row(query, tool_result, assistant, domain, "ranking", "BarChart")


def generate_threshold_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    threshold = metric["threshold"]
    if rng.random() < 0.5:
        value = nearby_value(rng, metric, threshold * rng.uniform(1.02, 1.35))
    else:
        value = nearby_value(rng, metric, threshold * rng.uniform(0.45, 0.96))
    if metric["decimals"] == 0:
        value = int(round(value))
    exceeded = value > threshold
    ratio = value / threshold if threshold else 0
    severity = "danger" if ratio >= 1.25 else "warning" if exceeded else "success"
    status = "exceeded" if exceeded else "normal"
    description = "Der gemessene Wert liegt über dem Grenzwert." if exceeded else "Der gemessene Wert liegt unter dem Grenzwert."
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "value": value,
        "threshold": threshold,
        "status": status,
        "severity": severity,
        "unit": metric["unit"],
    }
    query = query_from_templates(
        rng,
        [
            "Gab es eine Grenzwertüberschreitung bei {metric} in {location}?",
            "Prüfe den Grenzwert für {metric} in {location}.",
            "Ist {metric} in {location} über dem Grenzwert?",
        ],
        metric=metric["name"],
        location=location,
    )
    title = f"{metric['name']}-Grenzwert {'überschritten' if exceeded else 'eingehalten'}"
    assistant = render_notice(
        title,
        description,
        "warning" if exceeded else "info",
        [
            {"label": "Wert", "value": metric_value(value, metric["unit"]), "caption": str(year)},
            {"label": "Grenzwert", "value": metric_value(threshold, metric["unit"]), "caption": status},
        ],
    )
    return row(query, tool_result, assistant, domain, "threshold", "Notice")


def generate_percentage_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    percent_metric = metric if metric["unit"] == "%" else rng.choice([m for m in DOMAINS[domain] if m["unit"] == "%"] or [metric])
    value = max(0, min(100, value_for_metric(rng, percent_metric) if percent_metric["unit"] == "%" else rng.randint(5, 98)))
    tool_result = {
        "domain": domain,
        "metric": percent_metric["name"],
        "location": location,
        "year": year,
        "value": value,
        "max": 100,
        "unit": "%",
    }
    query = query_from_templates(
        rng,
        [
            "Wie hoch ist die Auslastung von {location}?",
            "Zeige den Prozentwert für {metric} in {location}.",
            "Wie groß ist der Anteil von {metric} in {location}?",
        ],
        metric=percent_metric["name"],
        location=location,
    )
    title = f"{percent_metric['name']} in {location}"
    description = f"Prozentwert für das Jahr {year}"
    assistant = render_metric_grid(
        title,
        [{"label": percent_metric["name"], "value": metric_value(value, "%"), "caption": description}],
        f"{percent_metric['name']} liegt bei {format_number(value)} von 100 Prozent.",
    )
    return row(query, tool_result, assistant, domain, "percentage", "MetricGrid")


def generate_table_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    rows = []
    for office in rng.sample(OFFICES, rng.randint(3, 8)):
        rows.append({"office": office, "value": value_for_metric(rng, metric), "unit": metric["unit"]})
    columns = [
        {"key": "office", "label": "Dienststelle"},
        {"key": "value", "label": metric["name"]},
        {"key": "unit", "label": "Einheit"},
    ]
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "columns": columns,
        "records": rows,
    }
    query = query_from_templates(
        rng,
        [
            "Liste die Werte nach Dienststelle auf.",
            "Zeige {metric} nach Dienststelle in {location}.",
            "Erstelle eine Tabelle für {metric} in {year}.",
        ],
        metric=metric["name"],
        location=location,
        year=year,
    )
    title = f"{metric['name']} nach Dienststelle {year}"
    return row(query, tool_result, render_data_table(title, rows), domain, "table", "DataTable")


def generate_multi_kpi_example(rng, domain=None):
    domain = domain or rng.choice(list(DOMAINS.keys()))
    location = rng.choice(LOCATIONS)
    year = rng.randint(2018, 2025)
    metrics = rng.sample(DOMAINS[domain], rng.randint(3, 5))
    kpis = [{"metric": metric["name"], "value": value_for_metric(rng, metric), "unit": metric["unit"]} for metric in metrics]
    tool_result = {
        "domain": domain,
        "location": location,
        "year": year,
        "kpis": kpis,
    }
    query = query_from_templates(
        rng,
        [
            "Erstelle eine Übersicht zu {domain} in {location}.",
            "Zeige die wichtigsten Kennzahlen für {domain} in {location}.",
            "Fasse {domain} für {location} als Dashboard zusammen.",
        ],
        domain=domain,
        location=location,
    )
    title = f"Übersicht zu {domain} in {location}"
    metrics_for_ui = [
        {"label": item["metric"], "value": metric_value(item["value"], item["unit"]), "caption": str(year)}
        for item in kpis
    ]
    return row(query, tool_result, render_metric_grid(title, metrics_for_ui), domain, "multi_kpi", "MetricGrid")


def generate_geo_values_example(rng, domain=None):
    domain, metric, location, year = choose_context(rng, domain)
    district_count = rng.randint(5, 10)
    values = [{"district": district, "value": value_for_metric(rng, metric)} for district in rng.sample(DISTRICTS, district_count)]
    tool_result = {
        "domain": domain,
        "metric": metric["name"],
        "location": location,
        "year": year,
        "unit": metric["unit"],
        "values": values,
    }
    query = query_from_templates(
        rng,
        [
            "Zeige mir {metric} je Stadtbezirk in {location}.",
            "Welche Werte gibt es für {metric} in den Bezirken?",
            "Visualisiere {metric} räumlich nach Stadtbezirk.",
        ],
        metric=metric["name"],
        location=location,
    )
    title = f"{metric['name']} nach Stadtbezirk in {location}"
    assistant = render_chart(title, "district", "value", values, f"Stadtbezirkswerte in {metric['unit']}.")
    return row(query, tool_result, assistant, domain, "geo_values", "BarChart")


GENERATORS = {
    "scalar": generate_scalar_example,
    "comparison": generate_comparison_example,
    "time_series_daily": generate_daily_series_example,
    "time_series_monthly": generate_monthly_series_example,
    "ranking": generate_ranking_example,
    "threshold": generate_threshold_example,
    "percentage": generate_percentage_example,
    "table": generate_table_example,
    "multi_kpi": generate_multi_kpi_example,
    "geo_values": generate_geo_values_example,
}


def validate_row(example):
    encoded = json.dumps(example, ensure_ascii=False)
    decoded = json.loads(encoded)
    assistant = decoded["messages"][2]["content"]
    if "```" in assistant:
        raise ValueError("assistant output contains markdown fence")
    if not assistant.startswith("root = Root(["):
        raise ValueError("assistant output does not start with OpenUI Lang root assignment")
    try:
        from app.openui_support import parse_openui_lang

        parse_openui_lang(assistant)
    except Exception as exc:
        raise ValueError(f"assistant output is not renderable OpenUI Lang: {exc}") from exc
    return encoded


def generate_dataset(num_samples, seed):
    rng = random.Random(seed)
    domains = list(DOMAINS.keys())
    rows = []
    seen = set()
    attempts = 0
    max_attempts = max(num_samples * 5, 100)
    while len(rows) < num_samples and attempts < max_attempts:
        index = len(rows)
        data_shape = DATA_SHAPES[index % len(DATA_SHAPES)]
        domain = domains[(index // len(DATA_SHAPES)) % len(domains)]
        example = GENERATORS[data_shape](rng, domain)
        key = (example["messages"][1]["content"], example["messages"][2]["content"])
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(example)
    if len(rows) != num_samples:
        raise RuntimeError(f"Could only generate {len(rows)} unique rows out of {num_samples}")
    return rows


def write_jsonl(rows, output):
    with open(output, "w", encoding="utf-8") as handle:
        for example in rows:
            handle.write(validate_row(example) + "\n")
    with open(output, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc


def print_summary(rows, output):
    domain_counts = Counter(row["metadata"]["domain"] for row in rows)
    shape_counts = Counter(row["metadata"]["data_shape"] for row in rows)
    component_counts = Counter(row["metadata"]["component"] for row in rows)
    print(f"generated samples: {len(rows)}")
    print(f"output path: {output}")
    print("count per domain:")
    for key in sorted(domain_counts):
        print(f"  {key}: {domain_counts[key]}")
    print("count per data shape:")
    for key in DATA_SHAPES:
        print(f"  {key}: {shape_counts[key]}")
    print("count per component:")
    for key in sorted(component_counts):
        print(f"  {key}: {component_counts[key]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic OpenUI SFT JSONL dataset.")
    parser.add_argument("--num-samples", type=int, default=20000, help="Number of examples to generate.")
    parser.add_argument("--output", default="openui_sft_dataset.jsonl", help="Output JSONL file path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    rows = generate_dataset(args.num_samples, args.seed)
    write_jsonl(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
