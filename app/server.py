"""Triton server for the ALIVE models."""

from typing import Any

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import DynamicBatcher, ModelConfig, QueuePolicy, TimeoutAction  # type: ignore
from pytriton.triton import Triton, TritonConfig  # type: ignore

from app.models import MODELS


def serve(
    http_port: int = 8000,
    **kwargs: Any,
) -> None:
    """Start the triton server."""
    triton_config = TritonConfig(
        http_port=http_port,
        grpc_port=kwargs.get("grpc_port", 8001),
        sagemaker_port=kwargs.get("sagemaker_port", 8080),
        sagemaker_safe_port_range=kwargs.get("sagemaker_safe_port_range", None),
        metrics_port=kwargs.get("metrics_port", 8002),
        allow_http=kwargs.get("allow_http", False),
        allow_grpc=kwargs.get("allow_grpc", True),
        allow_metrics=kwargs.get("allow_metrics", True),
        allow_sagemaker=kwargs.get("allow_sagemaker", True),
        grpc_root_cert=kwargs.get("grpc_root_cert", None),
        grpc_server_cert=kwargs.get("grpc_server_cert", None),
        grpc_server_key=kwargs.get("grpc_server_key", None),
        grpc_use_ssl=kwargs.get("grpc_use_ssl", False),
        grpc_use_ssl_mutual=kwargs.get("grpc_use_ssl_mutual", False),
        log_verbose=kwargs.get("log_verbose", None),
    )
    model_config = ModelConfig(
        batching=True,
        max_batch_size=128,
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=100000,  # 100 milliseconds
            preserve_ordering=False,
            default_queue_policy=QueuePolicy(
                timeout_action=TimeoutAction.REJECT,
                default_timeout_microseconds=1000000,  # 1 second
                allow_timeout_override=True,
                max_queue_size=100,
            ),
        ),
    )
    with Triton(config=triton_config) as triton:
        for model in MODELS:
            triton.bind(
                model_name=model.name,
                model_version=model.version,
                infer_func=model.infer_fn,
                inputs=model.inputs,
                outputs=model.outputs,
                config=model_config,
            )
        triton.serve()
