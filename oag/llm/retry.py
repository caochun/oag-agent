"""LLM API 重试封装。

集中处理 OpenAI API 的临时性错误重试、指数退避和 jitter，使主循环、
worker 和上下文摘要调用保持一致的失败处理方式。
"""

from __future__ import annotations

import logging
import random
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

BASE_DELAY = 0.5
MAX_DELAY = 32.0
DEFAULT_MAX_RETRIES = 5

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def call_llm_with_retry(client: OpenAI, max_retries: int = DEFAULT_MAX_RETRIES,
                         **kwargs):
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            last_error = e
            if e.status_code not in RETRYABLE_STATUS_CODES:
                raise
            delay = _backoff_delay(attempt)
            logger.warning("LLM API %d, retry %d/%d in %.1fs", e.status_code, attempt + 1, max_retries, delay)
        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            delay = _backoff_delay(attempt)
            logger.warning("LLM API %s, retry %d/%d in %.1fs", type(e).__name__, attempt + 1, max_retries, delay)
        except Exception:
            raise

        if attempt >= max_retries:
            break
        time.sleep(delay)

    raise last_error  # type: ignore[misc]


def _backoff_delay(attempt: int) -> float:
    base = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, 0.25 * base)
    return base + jitter
