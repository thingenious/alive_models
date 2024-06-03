"""Application entrypoint."""

import logging
import os
import sys
from pathlib import Path

try:
    from app.server import serve
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from app.server import serve

_KEY_PREFIX = "ALIVE_MODELS"


def _ports() -> tuple[int, int, int, int]:
    """Get the ports for the HTTP, gRPC, metrics, and SageMaker services."""
    http_port = int(os.environ.get(f"{_KEY_PREFIX}_HTTP_PORT", "8000"))
    grpc_port = int(os.environ.get(f"{_KEY_PREFIX}_GRPC_PORT", "8001"))
    metrics_port = int(os.environ.get(f"{_KEY_PREFIX}_METRICS_PORT", "8002"))
    sagemaker_port = int(os.environ.get(f"{_KEY_PREFIX}_SAGEMAKER_PORT", "8080"))
    return http_port, grpc_port, metrics_port, sagemaker_port


def _services() -> tuple[bool, bool, bool, bool]:
    """Get the service flags for the HTTP, gRPC, metrics, and SageMaker services."""
    allow_http = os.environ.get(f"{_KEY_PREFIX}_ALLOW_HTTP", "true").lower() != "false"
    allow_grpc = os.environ.get(f"{_KEY_PREFIX}_ALLOW_GRPC", "true").lower() != "false"
    allow_metrics = os.environ.get(f"{_KEY_PREFIX}_ALLOW_METRICS", "true").lower() != "false"
    allow_sagemaker = os.environ.get(f"{_KEY_PREFIX}_ALLOW_SAGEMAKER", "true").lower() != "false"
    return allow_http, allow_grpc, allow_metrics, allow_sagemaker


def _grpc_ssl(is_grpc_allowed: bool) -> tuple[bool | None, bool | None, str | None, str | None, str | None]:
    """Get the gRPC SSL configuration."""
    if not is_grpc_allowed:
        return None, None, None, None, None
    grpc_use_ssl: bool | None = os.environ.get(f"{_KEY_PREFIX}_GRPC_USE_SSL", "false").lower() == "true"
    grpc_use_ssl_mutual: bool | None = os.environ.get(f"{_KEY_PREFIX}_GRPC_USE_SSL_MUTUAL", "false").lower() == "true"
    grpc_root_cert = os.environ.get(f"{_KEY_PREFIX}_GRPC_ROOT_CERT", None)
    grpc_server_cert = os.environ.get(f"{_KEY_PREFIX}_GRPC_SERVER_CERT", None)
    grpc_server_key = os.environ.get(f"{_KEY_PREFIX}_GRPC_SERVER_KEY", None)
    if (
        grpc_root_cert
        and not Path(grpc_root_cert).exists()
        or str(grpc_root_cert).lower() in ("none", "null", "false", "")
    ):
        grpc_root_cert = None
    if (
        grpc_server_cert
        and not Path(grpc_server_cert).exists()
        or str(grpc_server_cert).lower() in ("none", "null", "false", "")
    ):
        grpc_server_cert = None
    if (
        grpc_server_key
        and not Path(grpc_server_key).exists()
        or str(grpc_server_key).lower() in ("none", "null", "false", "")
    ):
        grpc_server_key = None
    if not grpc_root_cert or not grpc_server_cert or not grpc_server_key:
        grpc_use_ssl = None
        grpc_use_ssl_mutual = None
        grpc_root_cert = None
        grpc_server_cert = None
    return grpc_use_ssl, grpc_use_ssl_mutual, grpc_root_cert, grpc_server_cert, grpc_server_key


def _sagemaker_safe_port_range(is_sagemaker_allowed: bool, sagemaker_port: int) -> str | None:
    """Get the safe port range for the SageMaker service."""
    if not is_sagemaker_allowed:
        return None
    sagemaker_safe_port_range = os.environ.get(f"{_KEY_PREFIX}_SAGEMAKER_SAFE_PORT_RANGE", None)
    if not sagemaker_safe_port_range:
        return None
    is_valid = True
    # expecting XXXX-YYYY format
    if not sagemaker_safe_port_range.count("-") == 1:
        is_valid = False
    start, end = sagemaker_safe_port_range.split("-")
    if not start.isdigit() or not end.isdigit():
        is_valid = False
    port_start = int(start)
    port_end = int(end)
    if port_start > port_end:
        is_valid = False
    if sagemaker_port < port_start or sagemaker_port > port_end:
        is_valid = False
    if not is_valid:
        return None
    return sagemaker_safe_port_range


def main() -> None:
    """Application entrypoint."""
    http_port, grpc_port, metrics_port, sagemaker_port = _ports()
    allow_http, allow_grpc, allow_metrics, allow_sagemaker = _services()
    sagemaker_safe_port_range = _sagemaker_safe_port_range(allow_sagemaker, sagemaker_port)
    grpc_use_ssl, grpc_use_ssl_mutual, grpc_root_cert, grpc_server_cert, grpc_server_key = _grpc_ssl(allow_grpc)
    is_debug = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes", "on")
    if not is_debug and "--debug" in sys.argv or "--log-verbose" in sys.argv:
        is_debug = True
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if not is_debug else logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    serve(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        sagemaker_port=sagemaker_port,
        sagemaker_safe_port_range=sagemaker_safe_port_range,
        allow_http=allow_http,
        allow_grpc=allow_grpc,
        allow_metrics=allow_metrics,
        allow_sagemaker=allow_sagemaker,
        grpc_use_ssl=grpc_use_ssl,
        grpc_use_ssl_mutual=grpc_use_ssl_mutual,
        grpc_root_cert=grpc_root_cert,
        grpc_server_cert=grpc_server_cert,
        grpc_server_key=grpc_server_key,
        log_verbose=4 if is_debug else None,
    )


if __name__ == "__main__":
    main()
