from __future__ import annotations

import json
from urllib import request


def _post_json(api_url: str, payload: dict, timeout: int) -> dict:
    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def predict_single(api_url: str, payload: dict) -> dict:
    return _post_json(api_url=api_url, payload=payload, timeout=20)


def predict_batch(api_url: str, records: list[dict]) -> dict:
    return _post_json(api_url=api_url, payload={"records": records}, timeout=60)
