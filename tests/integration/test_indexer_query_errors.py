"""Indexer query errors must not trip the circuit breaker.

OpenSearch answers a free-text search that expands past max_clause_count with HTTP 500
(search_phase_execution_exception / too_many_clauses). That is a bad request, not an outage:
it must not be retried, must not count towards opening the breaker, and a free-text alert
search should fall back to a bounded field list instead of failing.
"""

import json

import httpx
import pytest

from wazuh_mcp_server.api.wazuh_indexer import (
    FREE_TEXT_FALLBACK_FIELDS,
    IndexerQueryError,
    WazuhIndexerClient,
)
from wazuh_mcp_server.resilience import CircuitBreakerState

TOO_MANY_CLAUSES = {
    "error": {
        "root_cause": [{"type": "too_many_clauses", "reason": "maxClauseCount is set to 1024"}],
        "type": "search_phase_execution_exception",
        "reason": "all shards failed",
        "caused_by": {"type": "too_many_clauses", "reason": "maxClauseCount is set to 1024"},
    },
    "status": 500,
}

EMPTY_HITS = {"hits": {"hits": [], "total": {"value": 0}}}


async def _client_with(handler) -> WazuhIndexerClient:
    client = WazuhIndexerClient(host="localhost", username="u", password="p", verify_ssl=False)
    await client.initialize()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _free_text_clause(body: dict) -> dict:
    for clause in body["query"]["bool"]["must"]:
        if "simple_query_string" in clause:
            return clause["simple_query_string"]
    raise AssertionError(f"no simple_query_string clause in {body}")


class TestQueryErrorsDoNotTripBreaker:
    @pytest.mark.asyncio
    async def test_too_many_clauses_is_query_error_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(500, json=TOO_MANY_CLAUSES)

        client = await _client_with(handler)
        with pytest.raises(IndexerQueryError) as exc:
            await client._search("wazuh-alerts-*", {"match_all": {}})
        assert exc.value.reason == "too_many_clauses"
        assert len(calls) == 1  # not retried by tenacity

    @pytest.mark.asyncio
    async def test_repeated_query_errors_keep_breaker_closed(self):
        client = await _client_with(lambda request: httpx.Response(500, json=TOO_MANY_CLAUSES))
        threshold = client._circuit_breaker.config.failure_threshold
        for _ in range(threshold + 2):
            with pytest.raises(IndexerQueryError):
                await client._search("wazuh-alerts-*", {"match_all": {}})
        assert client._circuit_breaker.state == CircuitBreakerState.CLOSED
        assert client._circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_query_error_closes_half_open_trial(self):
        client = await _client_with(lambda request: httpx.Response(500, json=TOO_MANY_CLAUSES))
        cb = client._circuit_breaker
        cb.state = CircuitBreakerState.OPEN
        cb.failure_count = cb.config.failure_threshold
        cb.last_failure_time = 0  # recovery timeout long elapsed → next call is the trial
        with pytest.raises(IndexerQueryError):
            await client._search("wazuh-alerts-*", {"match_all": {}})
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_genuine_server_error_still_counts(self, monkeypatch):
        # Skip tenacity's backoff sleeps; the retries themselves still happen.
        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(WazuhIndexerClient._search.retry, "sleep", _no_sleep, raising=False)

        def handler(request):
            return httpx.Response(
                500,
                json={"error": {"root_cause": [{"type": "circuit_breaking_exception"}], "type": "x"}},
            )

        client = await _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._search("wazuh-alerts-*", {"match_all": {}})
        assert client._circuit_breaker.failure_count > 0

    @pytest.mark.asyncio
    async def test_aggregation_query_error_is_query_error(self):
        client = await _client_with(lambda request: httpx.Response(400, json=TOO_MANY_CLAUSES))
        with pytest.raises(IndexerQueryError):
            await client._execute_body("wazuh-alerts-*", {"size": 0, "query": {"match_all": {}}})


class TestFreeTextFallbackFields:
    @pytest.mark.asyncio
    async def test_free_text_retries_on_bounded_fields(self):
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if "fields" not in _free_text_clause(body):
                return httpx.Response(500, json=TOO_MANY_CLAUSES)
            return httpx.Response(200, json=EMPTY_HITS)

        client = await _client_with(handler)
        await client.get_alerts(query_text="200 OR 301 OR 302 OR 500", srcip="203.0.113.10")

        assert len(bodies) == 2
        assert "fields" not in _free_text_clause(bodies[0])  # first try keeps all-field search
        assert _free_text_clause(bodies[1])["fields"] == FREE_TEXT_FALLBACK_FIELDS
        assert client._circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_without_free_text_query_error_propagates(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(500, json=TOO_MANY_CLAUSES)

        client = await _client_with(handler)
        with pytest.raises(IndexerQueryError):
            await client.get_alerts(rule_id="31101")
        assert len(calls) == 1
