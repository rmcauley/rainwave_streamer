# Copied from Rainwave's code, until this can be merged with the rest of the project.

import pickle
import emcache
import config
from typing import Any

client: emcache.Client | None

in_memory: dict[bytes, Any] = {}


async def _build_emcache_client(host: str, port: int) -> emcache.Client:
    client = await emcache.create_client(
        [emcache.MemcachedHostAddress(host, port)],
        connection_timeout=config.memcache_connect_timeout,
        timeout=config.memcache_timeout,
    )
    # memcache doesn't test its connection on start, so we force a get
    await client.get(b"hello")
    return client


async def connect() -> None:
    global client
    global ratings_client

    if client:
        return
    client = await _build_emcache_client(config.memcache_host, config.memcache_port)


async def cache_set(key: str, value: Any, *, save_in_memory: bool = False) -> None:
    global client

    if not client:
        raise Exception("No memcache connection.")

    bytes_key = key.encode("utf-8")
    if save_in_memory or bytes_key in in_memory:
        in_memory[bytes_key] = value

    await client.set(bytes_key, pickle.dumps(value))


async def cache_get(key: str) -> Any:
    if not client:
        raise Exception("No memcache connection.")

    bytes_key = key.encode("utf-8")
    if bytes_key in in_memory:
        return in_memory[bytes_key]

    result = await client.get(bytes_key)
    if result is None:
        return None
    return pickle.loads(result.value)


async def cache_set_station(
    sid: int, key: str, value: Any, *, save_in_memory: bool = False
) -> None:
    await cache_set(f"sid{sid}_{key}", value, save_in_memory=save_in_memory)


async def cache_get_station(sid: int, key: str) -> Any:
    return cache_get(f"sid{sid}_{key}")
