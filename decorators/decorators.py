import asyncio
import functools
import logging
import time
import random
import traceback
import inspect
from typing import Callable, Optional, Dict, List, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field

from aiogram.types import Message, CallbackQuery
from aiogram.exceptions import (
    TelegramRetryAfter,
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramAPIError,
)
from sqlalchemy.exc import OperationalError

from config import Config, logger


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    next_attempt_allowed: float = 0.0


@dataclass
class RateLimitState:
    last_request_time: float = 0.0
    backoff_until: float = 0.0
    history: List[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_CIRCUIT_STATES: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
_RATE_LIMITS: Dict[str, RateLimitState] = defaultdict(RateLimitState)

_SIGNATURE_CACHE: Dict[Callable[..., Any], inspect.Signature] = {}


def _get_bound_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> inspect.BoundArguments:
    """Bind arguments to parameter names using cached signatures"""
    if func not in _SIGNATURE_CACHE:
        _SIGNATURE_CACHE[func] = inspect.signature(func)

    sig = _SIGNATURE_CACHE[func]
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def resilient_request(
    scope: str = "default",
    max_retries: int = Config.MAX_RETRIES,
    use_circuit_breaker: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wraps async operations with circuit breaking, exponential backoff, and automatic rate limiting"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if use_circuit_breaker:
                cb_state = _CIRCUIT_STATES[scope]
                if cb_state.is_open:
                    if time.time() < cb_state.next_attempt_allowed:
                        logger.warning(f"Circuit '{scope}' open. Blocking request.")
                        raise RuntimeError(
                            f"Service {scope} is temporarily unavailable."
                        )
                    else:
                        logger.info(
                            f"Circuit '{scope}' half-open. Testing connectivity."
                        )
                        cb_state.is_open = False

            chat_id = None
            try:
                bound_args = _get_bound_args(func, args, kwargs)
                chat_id = bound_args.arguments.get("chat_id")

                if not chat_id:
                    for arg in args:
                        if isinstance(arg, Message):
                            chat_id = arg.chat.id
                            break
                        elif isinstance(arg, CallbackQuery):
                            if arg.message and arg.message.chat:
                                chat_id = arg.message.chat.id
                            break
            except Exception:
                pass

            if chat_id and scope == "telegram":
                await _enforce_rate_limit(f"{scope}:{chat_id}")

            retry_count = 0
            backoff = Config.INITIAL_RETRY_DELAY

            while True:
                try:
                    result = await func(*args, **kwargs)

                    if use_circuit_breaker:
                        _CIRCUIT_STATES[scope] = CircuitBreakerState()

                    return result

                except TelegramRetryAfter as e:
                    wait_time = e.retry_after + 1
                    logger.warning(f"Telegram Rate Limit: Waiting {wait_time}s")

                    if chat_id:
                        await _apply_backoff(f"{scope}:{chat_id}", wait_time)

                    await asyncio.sleep(wait_time)
                    continue

                except TelegramBadRequest as e:
                    if "message is not modified" in e.message.lower():
                        return True
                    logger.error(f"Telegram Bad Request: {e}")
                    raise e

                except (TelegramNetworkError, TelegramAPIError, Exception) as e:
                    retry_count += 1

                    if use_circuit_breaker:
                        _record_circuit_failure(scope)

                    if retry_count > max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        raise e

                    sleep_time = backoff * (2 ** (retry_count - 1)) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} after {sleep_time:.2f}s due to: {e}"
                    )
                    await asyncio.sleep(sleep_time)

        return wrapper

    return decorator


def operation(
    name: str, notify_user: bool = False, validate_callback_prefix: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Standardizes logging, error handling, and input validation for user interaction handlers"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            msg = next((arg for arg in args if isinstance(arg, Message)), None)
            cb = next((arg for arg in args if isinstance(arg, CallbackQuery)), None)

            user_id = None

            if cb:
                user_id = cb.from_user.id
                if validate_callback_prefix and cb.data:
                    if not cb.data.startswith(validate_callback_prefix):
                        logger.warning(f"[{name}] Invalid Callback Prefix: {cb.data}")
                        await cb.answer()
                        return
            elif msg:
                user_id = msg.from_user.id if msg.from_user else None

            try:
                return await func(*args, **kwargs)

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"[{name}] CRASH after {elapsed:.2f}s [User:{user_id}]: {e}\n{traceback.format_exc()}"
                )

                if notify_user:
                    try:
                        error_text = "An unexpected error occurred."
                        if cb:
                            if isinstance(cb.message, Message):
                                await cb.message.answer(error_text)
                            await cb.answer("Error")
                        elif msg:
                            await msg.answer(error_text)
                    except Exception:
                        pass

            finally:
                pass

        return wrapper

    return decorator


def db_lock_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps async DB calls with retries for database lock errors (sqlite+aiosqlite)"""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_db_retries = 5
        retry_count = 0

        while retry_count < max_db_retries:
            try:
                return await func(*args, **kwargs)
            except OperationalError as e:
                if "database is locked" in str(e):
                    retry_count += 1
                    sleep_time = random.uniform(0.05, 0.25) * retry_count
                    logger.warning(
                        f"DB Locked ({func.__name__}). Retry {retry_count}/{max_db_retries} in {sleep_time:.3f}s"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise e
            except Exception as e:
                logger.error(f"DB Fatal Error in {func.__name__}: {e}")
                raise e

        # Re-raise specific error for locked db
        raise OperationalError(
            f"Database locked after {max_db_retries} retries in {func.__name__}",
            params={},
            orig=Exception("Maximum database retries exceeded"),
        )

    return wrapper


def cpu_bound(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to run a synchronous CPU-bound function in a thread pool"""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    return wrapper


async def _enforce_rate_limit(key: str) -> None:
    """Pauses execution to enforce minimum request intervals and active backoff penalties"""
    state = _RATE_LIMITS[key]

    async with state.lock:
        now = time.time()

        if now < state.backoff_until:
            wait_time = state.backoff_until - now
            if wait_time > 0.05:
                await asyncio.sleep(wait_time)
            now = time.time()

        if len(state.history) >= 2:
            recent_window = state.history[-5:]
            if len(recent_window) > 1:
                avg_interval = (recent_window[-1] - recent_window[0]) / (
                    len(recent_window) - 1
                )

                if avg_interval < Config.MIN_UPDATE_INTERVAL:
                    penalty = Config.MIN_UPDATE_INTERVAL - avg_interval
                    await asyncio.sleep(penalty)

        new_now = time.time()
        state.last_request_time = new_now
        state.history.append(new_now)

        if len(state.history) > 20:
            state.history = state.history[-20:]


async def _apply_backoff(key: str, seconds: float) -> None:
    """Apply penalty to a rate limit bucket"""
    state = _RATE_LIMITS[key]
    async with state.lock:
        state.backoff_until = time.time() + seconds


def _record_circuit_failure(scope: str) -> None:
    """Trip circuit breaker if needed"""
    state = _CIRCUIT_STATES[scope]
    state.failure_count += 1
    state.last_failure_time = time.time()

    if state.failure_count >= 3:
        state.is_open = True
        state.next_attempt_allowed = time.time() + 60
        logger.error(f"Circuit '{scope}' tripped. Pausing requests for 60s.")
