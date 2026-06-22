# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics export for the sglang-omni API server."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from starlette.routing import Match

_BUCKET_TTFB = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)
_BUCKET_E2E = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
)
_BUCKET_INTER_CHUNK = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)


class ServerMetricsCollector:
    """Custom collector that snapshots server health into Prometheus metrics."""

    def __init__(
        self,
        *,
        health_provider: Callable[[], dict[str, Any]],
        model_name_provider: Callable[[], str],
    ) -> None:
        self._health_provider = health_provider
        self._model_name_provider = model_name_provider

    def collect(self) -> Iterator[Metric]:
        health = self._health_provider()
        model_name = self._model_name_provider()

        up = GaugeMetricFamily(
            'sglang_omni_server_up',
            'Whether the sglang-omni server is healthy and accepting requests.',
        )
        up.add_metric([], 1 if health.get('running', False) else 0)
        yield up

        info_metric = GaugeMetricFamily(
            'sglang_omni_server_info',
            'Static server metadata.',
            labels=['model_name', 'entry_stage'],
        )
        info_metric.add_metric([model_name, str(health.get('entry_stage') or '')], 1)
        yield info_metric

        requests_metric = GaugeMetricFamily(
            'sglang_omni_server_requests',
            'Tracked request counts.',
            labels=['kind'],
        )
        requests_metric.add_metric(['total'], float(health.get('total_requests', 0) or 0))
        requests_metric.add_metric(
            ['pending_completions'],
            float(health.get('pending_completions', 0) or 0),
        )
        yield requests_metric

        stage_metric = GaugeMetricFamily(
            'sglang_omni_server_stage',
            'Registered stages for the coordinator.',
            labels=['stage'],
        )
        for stage in health.get('stages', []) or []:
            stage_metric.add_metric([str(stage)], 1)
        yield stage_metric

        state_metric = GaugeMetricFamily(
            'sglang_omni_server_request_states',
            'Tracked requests by coordinator state.',
            labels=['state'],
        )
        for state, count in (health.get('request_states') or {}).items():
            state_metric.add_metric([str(state)], float(count))
        yield state_metric


@dataclass
class ServerMetrics:
    registry: CollectorRegistry
    health_provider: Callable[[], dict[str, Any]]
    model_name_provider: Callable[[], str]
    app: FastAPI | None = None

    def __post_init__(self) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        self.registry.register(
            ServerMetricsCollector(
                health_provider=self.health_provider,
                model_name_provider=self.model_name_provider,
            )
        )

        self.http_requests_total = Counter(
            'sglang_omni_http_requests_total',
            'Total number of HTTP requests by endpoint and method.',
            labelnames=['endpoint', 'method'],
            registry=self.registry,
        )
        self.http_responses_total = Counter(
            'sglang_omni_http_responses_total',
            'Total number of HTTP responses by endpoint, method, and status code.',
            labelnames=['endpoint', 'method', 'status_code'],
            registry=self.registry,
        )
        self.http_requests_active = Gauge(
            'sglang_omni_http_requests_active',
            'Number of currently active HTTP requests.',
            labelnames=['endpoint', 'method'],
            registry=self.registry,
        )
        self.http_e2e_request_latency_seconds = Histogram(
            'sglang_omni_http_e2e_request_latency_seconds',
            'Histogram of end-to-end HTTP request latency in seconds.',
            labelnames=['endpoint', 'method'],
            buckets=_BUCKET_E2E,
            registry=self.registry,
        )
        self.http_time_to_first_byte_seconds = Histogram(
            'sglang_omni_http_time_to_first_byte_seconds',
            'Histogram of time to first response byte in seconds.',
            labelnames=['endpoint', 'method'],
            buckets=_BUCKET_TTFB,
            registry=self.registry,
        )
        self.http_inter_chunk_latency_seconds = Histogram(
            'sglang_omni_http_inter_chunk_latency_seconds',
            'Histogram of latency between streamed response chunks in seconds.',
            labelnames=['endpoint', 'method'],
            buckets=_BUCKET_INTER_CHUNK,
            registry=self.registry,
        )


        self.ss_inference_count_total = Counter(
            'ss_inference_count_total',
            'Number of inferences performed',
            registry=self.registry,
        )
        self.ss_inference_compute_preprocess_duration_s = Histogram(
            'ss_inference_compute_preprocess_duration_s',
            'Time spent preprocessing (s)',
            registry=self.registry,
        )
        self.ss_inference_compute_infer_duration_s = Histogram(
            'ss_inference_compute_infer_duration_s',
            'Time spent performing inference (s)',
            registry=self.registry,
        )
        self.ss_inference_compute_postprocess_duration_s = Histogram(
            'ss_inference_compute_postprocess_duration_s',
            'Time spent postprocessing (s)',
            registry=self.registry,
        )
        self.ss_inference_request_bytes = Histogram(
            'ss_inference_request_bytes',
            'Request size (bytes)',
            registry=self.registry,
        )
        self.ss_inference_response_bytes = Histogram(
            'ss_inference_response_bytes',
            'Response size (bytes)',
            registry=self.registry,
        )
        self.ss_active_requests = Gauge(
            'ss_active_requests',
            'Number of inference requests currently being processed',
            registry=self.registry,
        )
        self.ss_inference_latency_s = Histogram(
            'ss_inference_latency_s',
            'Latency of inference requests (s)',
            registry=self.registry,
        )

    def bind_app(self, app: FastAPI) -> None:
        self.app = app

    def resolve_path(self, scope: dict[str, Any]) -> str:
        app = self.app
        if app is None:
            return scope.get('path', '<unknown>')
        for route in app.routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return getattr(route, 'path', scope.get('path', '<unknown>'))
        return scope.get('path', '<unknown>')


class PrometheusMetricsMiddleware:
    def __init__(self, app, *, metrics: ServerMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        endpoint = self.metrics.resolve_path(scope)
        method = scope.get('method', 'UNKNOWN')
        labels = {'endpoint': endpoint, 'method': method}
        observe_ss_metrics = endpoint == '/v1/audio/speech' and method == 'POST'

        self.metrics.http_requests_total.labels(**labels).inc()
        self.metrics.http_requests_active.labels(**labels).inc()
        if observe_ss_metrics:
            self.metrics.ss_active_requests.inc()

        start = time.perf_counter()
        status_code = '500'
        finalized = False
        first_body_seen = False
        first_body_elapsed: float | None = None
        last_chunk_time: float | None = None
        request_bytes = 0
        response_bytes = 0

        async def receive_wrapper():
            nonlocal request_bytes
            message = await receive()
            if observe_ss_metrics and message.get('type') == 'http.request':
                request_bytes += len(message.get('body', b'') or b'')
            return message

        async def send_wrapper(message):
            nonlocal status_code, finalized, first_body_seen, first_body_elapsed, last_chunk_time, response_bytes

            if message['type'] == 'http.response.start':
                status_code = str(message['status'])
            elif message['type'] == 'http.response.body':
                now = time.perf_counter()
                body = message.get('body', b'') or b''
                if body:
                    if not first_body_seen:
                        elapsed = now - start
                        self.metrics.http_time_to_first_byte_seconds.labels(**labels).observe(
                            elapsed
                        )
                        if observe_ss_metrics:
                            first_body_elapsed = elapsed
                            self.metrics.ss_inference_compute_preprocess_duration_s.observe(
                                elapsed
                            )
                        first_body_seen = True
                    elif last_chunk_time is not None:
                        self.metrics.http_inter_chunk_latency_seconds.labels(**labels).observe(
                            now - last_chunk_time
                        )
                    last_chunk_time = now

                if not message.get('more_body', False):
                    self._finalize(
                        labels,
                        status_code,
                        start,
                        observe_ss_metrics=observe_ss_metrics,
                        first_body_elapsed=first_body_elapsed,
                        request_bytes=request_bytes,
                        response_bytes=response_bytes,
                    )
                    finalized = True

            if observe_ss_metrics and message['type'] == 'http.response.body':
                response_bytes += len(message.get('body', b'') or b'')
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            if not finalized:
                self._finalize(
                    labels,
                    '500',
                    start,
                    observe_ss_metrics=observe_ss_metrics,
                    first_body_elapsed=first_body_elapsed,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
            raise
        finally:
            if not finalized:
                self._finalize(
                    labels,
                    status_code,
                    start,
                    observe_ss_metrics=observe_ss_metrics,
                    first_body_elapsed=first_body_elapsed,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )

    def _finalize(
        self,
        labels: dict[str, str],
        status_code: str,
        start: float,
        *,
        observe_ss_metrics: bool,
        first_body_elapsed: float | None,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        self.metrics.http_responses_total.labels(
            endpoint=labels['endpoint'],
            method=labels['method'],
            status_code=status_code,
        ).inc()
        self.metrics.http_requests_active.labels(**labels).dec()
        total_elapsed = time.perf_counter() - start
        self.metrics.http_e2e_request_latency_seconds.labels(**labels).observe(
            total_elapsed
        )
        if observe_ss_metrics:
            self.metrics.ss_active_requests.dec()
            self.metrics.ss_inference_request_bytes.observe(float(request_bytes))
            self.metrics.ss_inference_response_bytes.observe(float(response_bytes))
            self.metrics.ss_inference_latency_s.observe(total_elapsed)
            self.metrics.ss_inference_compute_infer_duration_s.observe(
                max(total_elapsed - (first_body_elapsed or 0.0), 0.0)
            )
            self.metrics.ss_inference_compute_postprocess_duration_s.observe(0.0)
            if not status_code.startswith('5'):
                self.metrics.ss_inference_count_total.inc()


def build_server_metrics_registry(
    *,
    health_provider: Callable[[], dict[str, Any]],
    model_name_provider: Callable[[], str],
) -> ServerMetrics:
    registry = CollectorRegistry(auto_describe=True)
    GCCollector(registry=registry)
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    return ServerMetrics(
        registry=registry,
        health_provider=health_provider,
        model_name_provider=model_name_provider,
    )


def build_metrics_response(registry: CollectorRegistry) -> Response:
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
