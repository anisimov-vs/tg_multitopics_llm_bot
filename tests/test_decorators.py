import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from sqlalchemy.exc import OperationalError
from aiogram.exceptions import (
    TelegramRetryAfter,
    TelegramBadRequest,
    TelegramNetworkError,
)
from aiogram.types import Message, CallbackQuery, User

from decorators.decorators import (
    resilient_request,
    operation,
    db_lock_retry,
    cpu_bound,
    _CIRCUIT_STATES,
    _RATE_LIMITS,
    CircuitBreakerState,
)


@pytest.fixture(autouse=True)
def reset_decorator_state():
    """Reset global state dictionaries between tests."""
    _CIRCUIT_STATES.clear()
    _RATE_LIMITS.clear()
    yield


@pytest.fixture
def mock_sleep():
    """Mock asyncio.sleep to execute immediately."""
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_time():
    """Mock time.time to allow controlling flow."""
    with patch("time.time") as mock:
        mock.return_value = 1000.0
        yield mock


@pytest.mark.asyncio
async def test_resilient_request_success(mock_sleep):
    """Test normal execution without errors."""
    func = AsyncMock(return_value="Success")
    wrapped = resilient_request()(func)

    result = await wrapped("arg")
    assert result == "Success"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_resilient_request_retries(mock_sleep):
    """Test that it retries on generic exception."""
    # Fail twice, succeed third time
    func = AsyncMock(side_effect=[Exception("Fail 1"), Exception("Fail 2"), "Success"])
    wrapped = resilient_request(max_retries=3)(func)

    result = await wrapped()
    assert result == "Success"
    assert func.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_resilient_request_max_retries_exceeded(mock_sleep):
    """Test that it raises exception after max retries."""
    func = AsyncMock(side_effect=Exception("Persistent Fail"))
    wrapped = resilient_request(max_retries=2)(func)

    with pytest.raises(Exception, match="Persistent Fail"):
        await wrapped()

    assert func.call_count == 3


@pytest.mark.asyncio
async def test_telegram_retry_after(mock_sleep):
    """Test handling of TelegramRetryAfter (Rate Limit from API)."""
    retry_err = TelegramRetryAfter(method="post", message="wait", retry_after=5)
    func = AsyncMock(side_effect=[retry_err, "Success"])
    wrapped = resilient_request()(func)

    result = await wrapped()
    assert result == "Success"
    # Should sleep for retry_after + 1
    mock_sleep.assert_any_call(6)


@pytest.mark.asyncio
async def test_telegram_bad_request_not_modified():
    """Test that 'message is not modified' is ignored and returns True."""
    err = TelegramBadRequest(method="edit", message="Message is not modified: ...")
    func = AsyncMock(side_effect=err)
    wrapped = resilient_request()(func)

    result = await wrapped()
    assert result is True


@pytest.mark.asyncio
async def test_circuit_breaker_trips(mock_time, mock_sleep):
    """Test that the circuit breaker opens after failures."""
    func = AsyncMock(side_effect=TelegramNetworkError(method="test", message="Net err"))
    wrapped = resilient_request(scope="test_scope", use_circuit_breaker=True)(func)

    # 1. Fail 3 times to trip breaker
    with pytest.raises(TelegramNetworkError):
        await wrapped()

    # Verify state
    state = _CIRCUIT_STATES["test_scope"]
    assert state.is_open is True
    assert state.failure_count >= 3

    # 2. Next call should fail immediately with RuntimeError (Circuit Open)
    # We reset the func mock to ensure it's NOT called
    func.reset_mock()

    with pytest.raises(
        RuntimeError, match="Service test_scope is temporarily unavailable"
    ):
        await wrapped()

    func.assert_not_called()


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(mock_time, mock_sleep):
    """Test that circuit breaker attempts recovery after timeout."""
    # Setup open state manually
    _CIRCUIT_STATES["test_scope"] = CircuitBreakerState(
        is_open=True, next_attempt_allowed=1000.0 + 6
    )

    func = AsyncMock(return_value="Recovered")
    wrapped = resilient_request(scope="test_scope", use_circuit_breaker=True)(func)

    # Advance time past the cooldown
    mock_time.return_value = 1000.0 + 61

    result = await wrapped()
    assert result == "Recovered"
    assert _CIRCUIT_STATES["test_scope"].is_open is False


@pytest.mark.asyncio
async def test_operation_catches_exception():
    """Test that @operation suppresses exceptions and logs them."""
    func = AsyncMock(side_effect=Exception("Boom"))
    wrapped = operation(name="test_op")(func)

    # Should not raise
    result = await wrapped()
    assert result is None


@pytest.mark.asyncio
async def test_operation_notifies_user_on_error():
    """Test that @operation tries to send an error message to user."""
    msg = AsyncMock(spec=Message)
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = 123

    func = AsyncMock(side_effect=Exception("Boom"))
    wrapped = operation(name="test_op", notify_user=True)(func)

    await wrapped(msg)

    msg.answer.assert_called_with("An unexpected error occurred.")


@pytest.mark.asyncio
async def test_operation_validates_callback():
    """Test that invalid callback data prevents execution."""
    func = AsyncMock()
    wrapped = operation(name="test_cb", validate_callback_prefix="valid:")(func)

    cb = AsyncMock(spec=CallbackQuery)
    cb.data = "invalid:data"
    cb.from_user = MagicMock(spec=User)
    cb.from_user.id = 123
    cb.answer = AsyncMock()

    await wrapped(cb)

    # Func should NOT be called
    func.assert_not_called()
    # Should answer the callback to stop loading
    cb.answer.assert_called()


@pytest.mark.asyncio
async def test_db_lock_retry_success(mock_sleep):
    """Test retries on 'database is locked'."""
    # Fail twice with lock, then succeed
    locked_error = OperationalError("database is locked", params={}, orig=None)
    func = AsyncMock(side_effect=[locked_error, locked_error, "Success"])
    wrapped = db_lock_retry(func)

    result = await wrapped()
    assert result == "Success"
    assert func.call_count == 3


@pytest.mark.asyncio
async def test_db_lock_retry_fatal_error(mock_sleep):
    """Test that other OperationalErrors are raised immediately."""
    syntax_error = OperationalError("syntax error", params={}, orig=None)
    func = AsyncMock(side_effect=syntax_error)
    wrapped = db_lock_retry(func)

    with pytest.raises(OperationalError, match="syntax error"):
        await wrapped()

    assert func.call_count == 1


@pytest.mark.asyncio
async def test_db_lock_retry_exhausted(mock_sleep):
    """Test exhaustion of retries."""
    locked_error = OperationalError("database is locked", params={}, orig=None)
    func = AsyncMock(side_effect=locked_error)
    wrapped = db_lock_retry(func)

    with pytest.raises(OperationalError, match="Database locked after 5 retries"):
        await wrapped()

    assert func.call_count == 5


def sync_task(x):
    return x * 2


@pytest.mark.asyncio
async def test_cpu_bound():
    """Test that cpu_bound wraps sync function in thread executor."""
    wrapped = cpu_bound(sync_task)
    result = await wrapped(10)
    assert result == 20
