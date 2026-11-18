"""OpenAI-compatible client for vLLM, with retries and a disk cache so reruns never repeat a request."""
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from ts.repo import ROOT

DEFAULT_CACHE = ROOT / "runs" / "llm_cache.sqlite"


class ResponseCache:
    def __init__(self, path=DEFAULT_CACHE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=120)  # two runs may share the cache
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._conn.commit()

    def get(self, key):
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


def cache_key(model, prompt, params):
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMClient:
    def __init__(self, model, base_url="http://127.0.0.1:8000/v1", cache=None, retries=5, concurrency=32,
                 request_timeout=1800):
        self.model, self.retries, self.concurrency = model, retries, concurrency
        self.cache = cache if cache is not None else ResponseCache()
        # vLLM needs no key; the library requires a value. max_retries=0 because _request handles retries.
        self._client = OpenAI(base_url=base_url, api_key="unused", timeout=request_timeout, max_retries=0)

    def _send(self, prompt, api, system, params):
        if api == "completions":  # for models without a chat template, such as base Llama
            response = self._client.completions.create(model=self.model, prompt=prompt, **params)
            return response, response.choices[0].text
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        return response, response.choices[0].message.content

    def _request(self, prompt, api, system, params):
        last_error = None
        for attempt in range(self.retries):
            try:
                response, text = self._send(prompt, api, system, params)
                return {"text": text or "", "finish_reason": response.choices[0].finish_reason,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens}
            except Exception as error:  # server restarts and timeouts are retried; the last error is raised
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Request failed after {self.retries} attempts") from last_error

    def complete(self, prompt, api="chat", system=None, **params):
        # The first experiments had neither field, so they are left out of the key at their defaults.
        key_params = dict(params)
        if api != "chat":
            key_params["api"] = api
        if system:
            key_params["system"] = system
        key = cache_key(self.model, prompt, key_params)
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        result = self._request(prompt, api, system, params)
        self.cache.put(key, result)
        return {**result, "cached": False}

    def complete_many(self, prompts, api="chat", system=None, **params):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(tqdm(pool.map(lambda p: self.complete(p, api=api, system=system, **params), prompts),
                             total=len(prompts)))
