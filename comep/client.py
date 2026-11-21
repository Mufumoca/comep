from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class APIConfig:
    base_url: str
    model: str
    key_env: str = "OPENAI_API_KEY"
    max_calls: int = 100
    max_total_tokens: int = 100000
    max_output_tokens: int = 128
    timeout_seconds: float = 60
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    thinking: str | None = None
    max_retries: int = 0
    max_format_retries: int = 0
    cache_dir: str | None = None
    cache_namespace: str | None = None
    protocol_version: str = "v2.2"

    def __post_init__(self):
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be a credential-free HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("remote endpoints require HTTPS")
        if not self.model.strip():
            raise ValueError("model must be specified")
        if self.thinking not in (None, "enabled", "disabled"):
            raise ValueError("thinking must be enabled, disabled or unset")
        for value in (self.max_calls, self.max_total_tokens, self.max_output_tokens):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("positive integer API limits required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("positive timeout required")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not isinstance(self.max_format_retries, int) or isinstance(self.max_format_retries, bool) or self.max_format_retries < 0:
            raise ValueError("max_format_retries must be a nonnegative integer")
        prices = (self.input_price_per_million, self.output_price_per_million)
        if any(value is None for value in prices) and not all(value is None for value in prices):
            raise ValueError("supply both input and output prices or neither")
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in prices):
            raise ValueError("prices must be finite and nonnegative")


def uses_verified_magpie_mimo_reasoning_field(base_url: str, model: str) -> bool:
    return (
        base_url.rstrip("/").lower() == "http://127.0.0.1:3425/v1"
        and model.strip().lower() == "opencode-go-2/mimo-v2.6-flash"
    )


def uses_verified_openai_gpt6_luna_wire_format(base_url: str, model: str) -> bool:
    return (
        base_url.rstrip("/").lower() == "https://api.openai.com/v1"
        and model.strip().lower() == "gpt-6-luna"
    )


def build_chat_payload(config: APIConfig, messages: list[dict]) -> dict:
    if uses_verified_openai_gpt6_luna_wire_format(config.base_url, config.model):
        if config.thinking != "disabled":
            raise ValueError(
                "gpt-6-luna on https://api.openai.com/v1 requires explicit thinking='disabled' "
                "(mapped to temperature=0, max_completion_tokens, reasoning_effort='none')"
            )
        return {
            "model": config.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": config.max_output_tokens,
            "reasoning_effort": "none",
        }
    payload = {"model": config.model, "messages": messages, "temperature": 0, "max_tokens": config.max_output_tokens}
    if config.thinking is not None:
        if "qwen3" in config.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": config.thinking == "enabled"}
        elif uses_verified_magpie_mimo_reasoning_field(config.base_url, config.model):
            payload["reasoning"] = {"enabled": config.thinking == "enabled"}
        else:
            payload["thinking"] = {"type": config.thinking}
    return payload


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _load_key_from_dotenv(var_name: str) -> str | None:
    for candidate in (Path(".env.local"), Path(__file__).resolve().parent.parent / ".env.local"):
        if candidate.is_file():
            for raw_line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                k, v = line.split("=", 1)
                if k.strip() == var_name:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
    return None


def _validate_provider_body(body: object) -> tuple[str, int, int, str | None]:
    if not isinstance(body, dict):
        raise ValueError("API response must be a JSON object")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("provider must return a usage object")
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (prompt_tokens, completion_tokens)):
        raise ValueError("provider must return valid prompt_tokens and completion_tokens")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("API response has no valid choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "tool_calls":
        raise ValueError("API response returned unexpected tool_calls finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API response has no valid message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response has no text content")
    return content.strip(), prompt_tokens, completion_tokens, finish_reason


class ChatClient:
    def __init__(self, config, log_path):
        self.config = config
        self.key = os.environ.get(config.key_env) or _load_key_from_dotenv(config.key_env)
        if not self.key:
            raise ValueError(f"missing credential environment variable: {config.key_env}")
        self.log_path = Path(log_path)
        self.http_ledger_path = self.log_path.with_name(f"{self.log_path.stem}_http_attempts.jsonl")
        self.calls = 0
        self.http_requests = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.provider_reported_prompt_tokens = 0
        self.provider_reported_completion_tokens = 0
        self.usage_complete = True
        self.http_usage_complete = True
        self.halted = False
        self.purpose_calls = {}
        self.last_response = None
        self.opener = urllib.request.build_opener(NoRedirect())
        self.sync_http_ledger()

    def sync_http_ledger(self):
        """Reconstruct non-rollbackable live HTTP request & usage counters from two-phase append-only ledger."""
        if not self.http_ledger_path.is_file():
            return
        started_ids = []
        started_set = set()
        finished_map = {}
        for raw_line in self.http_ledger_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            rec = json.loads(raw_line)
            aid = rec.get("attempt_id") or f"legacy_{rec.get('http_request', len(started_ids) + 1)}"
            phase = rec.get("phase", "finished")
            if phase == "started":
                if aid not in started_set:
                    started_set.add(aid)
                    started_ids.append(aid)
            elif phase == "finished":
                if aid not in started_set:
                    started_set.add(aid)
                    started_ids.append(aid)
                finished_map[aid] = rec
        pt_sum = 0
        ct_sum = 0
        complete_flag = True
        for aid in started_ids:
            f_rec = finished_map.get(aid)
            if f_rec is not None and f_rec.get("usage_complete", False):
                pt_sum += int(f_rec.get("prompt_tokens", 0))
                ct_sum += int(f_rec.get("completion_tokens", 0))
            else:
                complete_flag = False
        self.http_requests = len(started_ids)
        self.provider_reported_prompt_tokens = pt_sum
        self.provider_reported_completion_tokens = ct_sum
        self.http_usage_complete = complete_flag

    def _start_http_attempt(self, *, purpose: str, attempt: int) -> str:
        self.http_requests += 1
        attempt_id = f"c{self.calls}_a{attempt}_{self.http_requests}_{time.time_ns()}"
        entry = {
            "phase": "started",
            "attempt_id": attempt_id,
            "http_request": self.http_requests,
            "logical_call": self.calls,
            "purpose": purpose,
            "attempt": attempt,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.http_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.http_ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, allow_nan=False) + "\n")
        return attempt_id

    def _finish_http_attempt(
        self,
        *,
        attempt_id: str,
        purpose: str,
        attempt: int,
        status: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        finish_reason: str | None = None,
        error_type: str | None = None,
    ):
        has_usage = (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        )
        if has_usage:
            self.provider_reported_prompt_tokens += prompt_tokens
            self.provider_reported_completion_tokens += completion_tokens
        else:
            self.http_usage_complete = False
        entry = {
            "phase": "finished",
            "attempt_id": attempt_id,
            "http_request": self.http_requests,
            "logical_call": self.calls,
            "purpose": purpose,
            "attempt": attempt,
            "status": status,
            "usage_complete": has_usage,
            "prompt_tokens": prompt_tokens if has_usage else None,
            "completion_tokens": completion_tokens if has_usage else None,
            "finish_reason": finish_reason,
            "error_type": error_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.http_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.http_ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, allow_nan=False) + "\n")

    def summary(self):
        logical_price = None
        provider_reported_price = None
        if self.usage_complete and self.config.input_price_per_million is not None:
            logical_price = (self.prompt_tokens * self.config.input_price_per_million + self.completion_tokens * self.config.output_price_per_million) / 1_000_000
        if self.http_usage_complete and self.config.input_price_per_million is not None:
            provider_reported_price = (
                self.provider_reported_prompt_tokens * self.config.input_price_per_million
                + self.provider_reported_completion_tokens * self.config.output_price_per_million
            ) / 1_000_000
        return {
            "attempted_calls": self.calls,
            "http_requests": self.http_requests,
            "cache_hits": self.cache_hits,
            "calls_by_purpose": dict(self.purpose_calls),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider_reported_prompt_tokens": self.provider_reported_prompt_tokens,
            "provider_reported_completion_tokens": self.provider_reported_completion_tokens,
            "usage_complete": self.usage_complete,
            "http_usage_complete": self.http_usage_complete,
            "estimated_cost_usd": logical_price,
            "provider_reported_cost_usd": provider_reported_price,
        }

    def _compute_cache_key(self, payload: dict, purpose: str, request_context: dict | None = None) -> str:
        canonical = {
            "protocol_version": self.config.protocol_version,
            "replicate_id": self.config.cache_namespace or "default",
            "provider_base_url": self.config.base_url.rstrip("/"),
            "model": self.config.model,
            "thinking": self.config.thinking,
            "purpose": purpose,
            "request_context": request_context or {},
            "payload": payload,
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def complete(self, messages, purpose="action", request_context: dict | None = None):
        if self.halted:
            raise RuntimeError("client halted after a failed or over-budget request")
        reserve = sum(len(message["content"].encode("utf-8")) + 64 for message in messages) + self.config.max_output_tokens + 64
        current_tokens = max(
            self.prompt_tokens + self.completion_tokens,
            self.provider_reported_prompt_tokens + self.provider_reported_completion_tokens,
        )
        if self.calls >= self.config.max_calls or current_tokens + reserve > self.config.max_total_tokens:
            raise BudgetExceeded("API call/token reservation limit reached before request")
        self.calls += 1
        self.purpose_calls[purpose] = self.purpose_calls.get(purpose, 0) + 1
        started = time.perf_counter()
        self.last_response = None
        event = {"call": self.calls, "purpose": purpose, "model": self.config.model, "reserved_tokens_estimate": reserve,
                 "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat()}
        if request_context:
            event["request_context"] = request_context
        usage_recorded = False
        try:
            payload = build_chat_payload(self.config, messages)
            encoded_payload = json.dumps(payload).encode()
            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "User-Agent": "CoMeP-Client/2.2",
            }
            url = self.config.base_url.rstrip("/") + "/chat/completions"
            raw = None
            body = None
            cache_hit = False

            def _do_live_request():
                nonlocal raw, body
                for attempt in range(self.config.max_retries + 1):
                    live_tokens = max(
                        self.prompt_tokens + self.completion_tokens,
                        self.provider_reported_prompt_tokens + self.provider_reported_completion_tokens,
                    )
                    if live_tokens + reserve > self.config.max_total_tokens:
                        raise BudgetExceeded("API token limit reached during retry loop")
                    attempt_id = self._start_http_attempt(purpose=purpose, attempt=attempt)
                    try:
                        request = urllib.request.Request(url, data=encoded_payload, headers=headers, method="POST")
                        with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                            raw = response.read(8 * 1024 * 1024 + 1)
                        if len(raw) > 8 * 1024 * 1024:
                            self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="oversize_response", prompt_tokens=None, completion_tokens=None, error_type="ValueError")
                            raise ValueError("API response exceeds 8 MiB")
                        try:
                            cand_body = json.loads(raw)
                        except Exception as parse_err:
                            self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="invalid_json_response", prompt_tokens=None, completion_tokens=None, error_type=type(parse_err).__name__)
                            raise
                        cand_usage = cand_body.get("usage") if isinstance(cand_body, dict) else {}
                        cand_pt = cand_usage.get("prompt_tokens") if isinstance(cand_usage, dict) else None
                        cand_ct = cand_usage.get("completion_tokens") if isinstance(cand_usage, dict) else None
                        cand_choices = cand_body.get("choices") if isinstance(cand_body, dict) else None
                        cand_msg = cand_choices[0].get("message", {}) if isinstance(cand_choices, list) and cand_choices and isinstance(cand_choices[0], dict) else {}
                        cand_fr = cand_choices[0].get("finish_reason") if isinstance(cand_choices, list) and cand_choices and isinstance(cand_choices[0], dict) else None
                        cand_content = cand_msg.get("content") if isinstance(cand_msg, dict) else None
                        if attempt < self.config.max_retries and (cand_fr == "tool_calls" or not isinstance(cand_content, str) or not cand_content.strip()):
                            self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="discarded_invalid_reply", prompt_tokens=cand_pt, completion_tokens=cand_ct, finish_reason=cand_fr)
                            time.sleep(min(1.0 * (attempt + 1), 6.0))
                            continue
                        try:
                            _validate_provider_body(cand_body)
                        except Exception as val_err:
                            self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="invalid_provider_body", prompt_tokens=cand_pt, completion_tokens=cand_ct, finish_reason=cand_fr, error_type=type(val_err).__name__)
                            raise
                        self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="ok", prompt_tokens=cand_pt, completion_tokens=cand_ct, finish_reason=cand_fr)
                        body = cand_body
                        if attempt > 0:
                            event["retries"] = attempt
                        break
                    except urllib.error.HTTPError as http_err:
                        self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status=f"http_{http_err.code}", prompt_tokens=None, completion_tokens=None, error_type="HTTPError")
                        if attempt < self.config.max_retries and http_err.code in (429, 500, 502, 503, 504):
                            time.sleep(min(1.5 * (2 ** attempt), 16.0))
                            continue
                        raise
                    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as net_err:
                        self._finish_http_attempt(attempt_id=attempt_id, purpose=purpose, attempt=attempt, status="network_error", prompt_tokens=None, completion_tokens=None, error_type=type(net_err).__name__)
                        if attempt < self.config.max_retries:
                            time.sleep(min(1.5 * (2 ** attempt), 16.0))
                            continue
                        raise
                if body is None:
                    body = json.loads(raw)

            if self.config.cache_dir:
                cache_root = Path(self.config.cache_dir)
                locks_dir = cache_root / "locks"
                entries_dir = cache_root / "entries"
                locks_dir.mkdir(parents=True, exist_ok=True)
                entries_dir.mkdir(parents=True, exist_ok=True)
                ckey = self._compute_cache_key(payload, purpose, request_context)
                event["cache_key"] = ckey[:16]
                lock_file = locks_dir / f"{ckey[:24]}.lock"
                entry_file = entries_dir / f"{ckey}.json"
                with lock_file.open("a+", encoding="utf-8") as lfh:
                    fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
                    try:
                        if entry_file.is_file():
                            cached_candidate = json.loads(entry_file.read_text(encoding="utf-8"))
                            _validate_provider_body(cached_candidate)
                            body = cached_candidate
                            cache_hit = True
                            self.cache_hits += 1
                            event["cache_hit"] = True
                        else:
                            _do_live_request()
                            event["cache_hit"] = False
                            # Validate full provider structure, usage, and finish_reason BEFORE atomic cache commit
                            _validate_provider_body(body)
                            tmp_entry = entries_dir / f"{ckey}.{os.getpid()}.tmp"
                            tmp_entry.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                            os.replace(tmp_entry, entry_file)
                    finally:
                        fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)
            else:
                _do_live_request()

            content, prompt_tokens, completion_tokens, finish_reason = _validate_provider_body(body)
            event["response_model"] = body.get("model")
            event["system_fingerprint"] = body.get("system_fingerprint")
            usage = body.get("usage", {})
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            usage_recorded = True
            event["usage"] = usage
            event["finish_reason"] = finish_reason
            event["reply"] = body["choices"][0]["message"]["content"]
            self.last_response = {key: event[key] for key in ("response_model", "system_fingerprint", "finish_reason", "reply")}
            if (
                self.prompt_tokens + self.completion_tokens > self.config.max_total_tokens
                or self.provider_reported_prompt_tokens + self.provider_reported_completion_tokens > self.config.max_total_tokens
            ):
                raise BudgetExceeded("provider-reported usage exceeded configured limit; stopping")
            event["status"] = "ok"
            return content
        except Exception as error:
            self.halted = True
            self.usage_complete = self.usage_complete and usage_recorded
            event["status"] = "error"
            event["error_type"] = type(error).__name__
            raise
        finally:
            event["seconds"] = time.perf_counter() - started
            event["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, allow_nan=False) + "\n")


def complete_and_validate(
    client: ChatClient,
    messages: list[dict],
    purpose: str,
    validator,
    correction_instruction: str,
    max_format_retries: int | None = None,
    request_context: dict | None = None,
) -> tuple[str, object, dict]:
    """Execute client.complete with up to max_format_retries format-only retries before falling back.

    Does not catch BudgetExceeded or transport/auth errors.
    Includes format_attempt_index in request_context so consecutive identical malformed outputs
    do not collide in the request cache.
    Returns (final_reply, parsed_value_or_None, retry_meta).
    """
    limit = client.config.max_format_retries if max_format_retries is None else int(max_format_retries)
    cur_messages = list(messages)
    format_errors = []
    last_reply = ""
    base_ctx = dict(request_context) if request_context else {}
    for fmt_attempt in range(limit + 1):
        call_purpose = purpose if fmt_attempt == 0 else f"{purpose}_format_retry"
        attempt_ctx = {**base_ctx, "format_attempt_index": fmt_attempt} if (base_ctx or fmt_attempt > 0 or client.config.cache_dir) else None
        reply = client.complete(cur_messages, call_purpose, request_context=attempt_ctx)
        last_reply = reply
        try:
            parsed = validator(reply)
            return reply, parsed, {
                "format_attempts": fmt_attempt + 1,
                "format_retries": fmt_attempt,
                "format_errors": format_errors,
                "format_fallback": False,
            }
        except (ValueError, json.JSONDecodeError) as fmt_err:
            format_errors.append(f"attempt_{fmt_attempt}:{type(fmt_err).__name__}:{fmt_err}")
            if fmt_attempt < limit:
                cur_messages = list(messages) + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"[Format retry {fmt_attempt + 1}/{limit}] {correction_instruction}"},
                ]
    return last_reply, None, {
        "format_attempts": limit + 1,
        "format_retries": limit,
        "format_errors": format_errors,
        "format_fallback": True,
    }
