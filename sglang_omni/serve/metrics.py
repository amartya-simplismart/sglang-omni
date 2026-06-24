# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics export for the sglang-omni API server."""

from __future__ import annotations

import time
from collections.abc import Callable
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
from starlette.routing import Match

_MILLISECONDS_PER_SECOND = 1000.0


class PrometheusLogger:
    def __init__(self, *, registry: CollectorRegistry):
        from prometheus_client import Counter, Gauge, Histogram

        self.inference_count = Counter(
            "ss_inference_count", "Number of inferences performed", registry=registry
        )

        self.compute_preprocess_duration_s = Histogram(
            "ss_inference_compute_preprocess_duration_s",
            "Time spent preprocessing (s)",
            registry=registry,
        )

        self.compute_infer_duration_s = Histogram(
            "ss_inference_compute_infer_duration_s",
            "Time spent performing inference (s)",
            registry=registry,
        )

        self.compute_postprocess_duration_s = Histogram(
            "ss_inference_compute_postprocess_duration_s",
            "Time spent postprocessing (s)",
            registry=registry,
        )

        self.request_bytes = Histogram(
            "ss_inference_request_bytes", "Request size (bytes)", registry=registry
        )

        self.response_bytes = Histogram(
            "ss_inference_response_bytes", "Response size (bytes)", registry=registry
        )

        self.num_active_requests = Gauge(
            "ss_active_requests",
            "Number of inference requests currently being processed",
            registry=registry,
        )

        self.inference_latency = Histogram(
            "ss_inference_latency_s", "Latency of inference requests (s)", registry=registry
        )

    def process(self, key: str, value: float = 1.0) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        metric_attr = getattr(self, key)
        if isinstance(metric_attr, Histogram):
            metric_attr.observe(value)
        elif isinstance(metric_attr, Counter):
            metric_attr.inc(value)
        elif isinstance(metric_attr, Gauge):
            metric_attr.set(value)


@dataclass
class ServerMetrics:
    registry: CollectorRegistry
    health_provider: Callable[[], dict[str, Any]]
    model_name_provider: Callable[[], str]
    app: FastAPI | None = None

    def __post_init__(self) -> None:
        self.prometheus_logger = PrometheusLogger(registry=self.registry)

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
        observe_ss_metrics = endpoint == '/v1/audio/speech' and method == 'POST'

        if observe_ss_metrics:
            self.metrics.prometheus_logger.num_active_requests.inc()

        start = time.perf_counter()
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
            nonlocal finalized, first_body_seen, first_body_elapsed, last_chunk_time, response_bytes

            if message['type'] == 'http.response.body':
                now = time.perf_counter()
                body = message.get('body', b'') or b''
                if body:
                    if not first_body_seen:
                        elapsed = now - start
                        if observe_ss_metrics:
                            first_body_elapsed = elapsed
                            self.metrics.prometheus_logger.process(
                                'compute_preprocess_duration_s',
                                elapsed * _MILLISECONDS_PER_SECOND
                            )
                        first_body_seen = True
                    elif last_chunk_time is not None and observe_ss_metrics:
                        self.metrics.prometheus_logger.process(
                            'compute_postprocess_duration_s',
                            (now - last_chunk_time) * _MILLISECONDS_PER_SECOND
                        )
                    last_chunk_time = now

                if observe_ss_metrics:
                    response_bytes += len(body)

                if not message.get('more_body', False):
                    self._finalize(
                        start,
                        observe_ss_metrics=observe_ss_metrics,
                        first_body_elapsed=first_body_elapsed,
                        request_bytes=request_bytes,
                        response_bytes=response_bytes,
                    )
                    finalized = True

            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            if not finalized:
                self._finalize(
                    start,
                    observe_ss_metrics=observe_ss_metrics,
                    first_body_elapsed=first_body_elapsed,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
                finalized = True
            raise
        finally:
            if not finalized:
                self._finalize(
                    start,
                    observe_ss_metrics=observe_ss_metrics,
                    first_body_elapsed=first_body_elapsed,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )

    def _finalize(
        self,
        start: float,
        *,
        observe_ss_metrics: bool,
        first_body_elapsed: float | None,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        total_elapsed = time.perf_counter() - start
        if observe_ss_metrics:
            self.metrics.prometheus_logger.num_active_requests.dec()
            self.metrics.prometheus_logger.process('request_bytes', float(request_bytes))
            self.metrics.prometheus_logger.process('response_bytes', float(response_bytes))
            self.metrics.prometheus_logger.process(
                'inference_latency', total_elapsed * _MILLISECONDS_PER_SECOND
            )
            self.metrics.prometheus_logger.process(
                'compute_infer_duration_s',
                max(total_elapsed - (first_body_elapsed or 0.0), 0.0)
                * _MILLISECONDS_PER_SECOND,
            )
            self.metrics.prometheus_logger.process('inference_count')


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
