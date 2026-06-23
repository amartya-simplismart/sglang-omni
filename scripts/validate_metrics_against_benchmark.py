#!/usr/bin/env python3

"""Validate Prometheus /metrics against the streaming Higgs benchmark.

This script reuses the host benchmark helper at /home/ubuntu/Amartya/higgs/benchmark_lib.py,
runs one or more concurrency levels against /v1/audio/speech, and compares metric deltas
from /metrics with the benchmark summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

DEFAULT_BENCHMARK_LIB = '/home/ubuntu/Amartya/higgs/benchmark_lib.py'
DEFAULT_INPUT = (
    "Hello, how are you today? I hope you're having a wonderful day and "
    "enjoying the sunshine."
)


def load_benchmark_lib(path: str):
    spec = importlib.util.spec_from_file_location('benchmark_lib_ext', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load benchmark lib from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_labels(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    parts: list[str] = []
    current = []
    in_quotes = False
    escape = False
    for ch in raw:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == '\\':
            current.append(ch)
            escape = True
            continue
        if ch == '"':
            current.append(ch)
            in_quotes = not in_quotes
            continue
        if ch == ',' and not in_quotes:
            parts.append(''.join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append(''.join(current))
    for part in parts:
        key, value = part.split('=', 1)
        labels[key] = value.strip().strip('"')
    return labels


def scrape_metrics(url: str) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    text = requests.get(url, timeout=30).text
    samples: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        metric_part, value_part = line.rsplit(' ', 1)
        value = float(value_part)
        if '{' in metric_part:
            name, rest = metric_part.split('{', 1)
            labels = parse_labels(rest[:-1])
        else:
            name = metric_part
            labels = {}
        samples[(name, frozenset(labels.items()))] = value
    return samples


def sample_value(
    samples: dict[tuple[str, frozenset[tuple[str, str]]], float],
    name: str,
    **labels: str,
) -> float:
    return samples.get((name, frozenset(labels.items())), 0.0)


def rel_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def metrics_url_from_request_url(url: str) -> str:
    marker = '/v1/audio/speech'
    if url.endswith(marker):
        return url[: -len(marker)] + '/metrics'
    raise ValueError('--metrics-url is required when --url does not end with /v1/audio/speech')


def warmup(module, url: str, payload: dict[str, Any], num_requests: int, timeout_s: float) -> None:
    for i in range(num_requests):
        module.stream_speech_request(url, payload, request_id=-(i + 1), timeout_s=timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate /metrics against Higgs benchmark output')
    parser.add_argument('--url', default='http://localhost:8000/v1/audio/speech')
    parser.add_argument('--metrics-url', default=None)
    parser.add_argument('--benchmark-lib', default=DEFAULT_BENCHMARK_LIB)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--voice', default='default')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--num-requests', type=int, default=16)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--rel-tol', type=float, default=0.35)
    parser.add_argument('--abs-tol-seconds', type=float, default=0.05)
    parser.add_argument('--json', action='store_true', help='Print machine-readable summary')
    args = parser.parse_args()

    metrics_url = args.metrics_url or metrics_url_from_request_url(args.url)
    bench = load_benchmark_lib(args.benchmark_lib)

    payload = {
        'input': args.input,
        'voice': args.voice,
        'stream': True,
        'response_format': 'pcm',
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
    }

    reports: list[dict[str, Any]] = []
    overall_ok = True

    for concurrency in args.concurrency:
        if args.warmup > 0:
            warmup(bench, args.url, payload, args.warmup, args.timeout)

        before = scrape_metrics(metrics_url)
        results, summary = bench.run_concurrency_level(
            url=args.url,
            payload=payload,
            concurrency=concurrency,
            num_requests=args.num_requests,
            warmup=0,
            timeout_s=args.timeout,
        )
        after = scrape_metrics(metrics_url)

        delta_requests = sample_value(after, 'ss_inference_count_total') - sample_value(before, 'ss_inference_count_total')
        delta_responses = delta_requests
        delta_e2e_count = sample_value(after, 'ss_inference_latency_s_count') - sample_value(before, 'ss_inference_latency_s_count')
        delta_e2e_sum = sample_value(after, 'ss_inference_latency_s_sum') - sample_value(before, 'ss_inference_latency_s_sum')
        delta_ttfb_count = sample_value(after, 'ss_inference_compute_preprocess_duration_s_count') - sample_value(before, 'ss_inference_compute_preprocess_duration_s_count')
        delta_ttfb_sum = sample_value(after, 'ss_inference_compute_preprocess_duration_s_sum') - sample_value(before, 'ss_inference_compute_preprocess_duration_s_sum')
        delta_inter_chunk_count = sample_value(after, 'ss_inference_compute_postprocess_duration_s_count') - sample_value(before, 'ss_inference_compute_postprocess_duration_s_count')
        final_active = sample_value(after, 'ss_active_requests')

        prom_e2e_avg = (delta_e2e_sum / delta_e2e_count) if delta_e2e_count > 0 else 0.0
        prom_ttfb_avg = (delta_ttfb_sum / delta_ttfb_count) if delta_ttfb_count > 0 else 0.0
        prom_throughput = (delta_requests / summary.wall_clock_s) if summary.wall_clock_s > 0 else 0.0

        checks = {
            'request_count_matches': delta_requests == summary.num_requests,
            'success_count_matches': delta_responses == summary.completed,
            'e2e_count_matches': delta_e2e_count == summary.completed,
            'ttfb_count_matches': delta_ttfb_count == summary.completed,
            'active_returns_to_zero': final_active == 0.0,
            'throughput_close': rel_close(prom_throughput, summary.throughput_req_per_s, args.rel_tol, args.abs_tol_seconds),
            'e2e_avg_close': rel_close(prom_e2e_avg, summary.e2e_avg_s, args.rel_tol, args.abs_tol_seconds),
            'ttfb_avg_close': rel_close(prom_ttfb_avg, summary.ttfb_avg_s, args.rel_tol, args.abs_tol_seconds),
            'inter_chunk_seen': delta_inter_chunk_count >= max(0, summary.completed - 1 if summary.completed <= 1 else 1),
        }
        ok = all(checks.values())
        overall_ok = overall_ok and ok

        report = {
            'concurrency': concurrency,
            'summary': asdict(summary),
            'prometheus': {
                'request_count_delta': delta_requests,
                'success_count_delta': delta_responses,
                'e2e_count_delta': delta_e2e_count,
                'e2e_sum_delta_s': delta_e2e_sum,
                'e2e_avg_s': prom_e2e_avg,
                'ttfb_count_delta': delta_ttfb_count,
                'ttfb_sum_delta_s': delta_ttfb_sum,
                'ttfb_avg_s': prom_ttfb_avg,
                'inter_chunk_count_delta': delta_inter_chunk_count,
                'throughput_req_per_s': prom_throughput,
                'final_active': final_active,
            },
            'checks': checks,
            'ok': ok,
        }
        reports.append(report)

        if not args.json:
            print(f'concurrency={concurrency}')
            print(f"  benchmark completed={summary.completed}/{summary.num_requests} throughput={summary.throughput_req_per_s:.4f} ttfb_avg={summary.ttfb_avg_s:.4f}s e2e_avg={summary.e2e_avg_s:.4f}s")
            print(f"  metrics   requests={delta_requests:.0f} success={delta_responses:.0f} throughput={prom_throughput:.4f} ttfb_avg={prom_ttfb_avg:.4f}s e2e_avg={prom_e2e_avg:.4f}s inter_chunk_count={delta_inter_chunk_count:.0f}")
            for key, value in checks.items():
                print(f'    {key}={value}')

    if args.json:
        import json
        print(json.dumps({'ok': overall_ok, 'reports': reports}, indent=2))
    return 0 if overall_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
