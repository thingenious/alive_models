"""Triton server for the ALIVE models."""

from typing import Any

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import DynamicBatcher, ModelConfig, QueuePolicy, TimeoutAction  # type: ignore
from pytriton.triton import Triton, TritonConfig  # type: ignore

from app.models import MODELS


def serve(
    **kwargs: Any,
) -> None:
    """Start the triton server.

    Parameters
    ----------
    http_port : int, optional
        The HTTP port to use, by default 8000.
    grpc_port : int, optional
        The gRPC port to use, by default 8001.
    sagemaker_port : int, optional
        The SageMaker port to use, by default 8080.
    sagemaker_safe_port_range : Optional[Tuple[int, int]], optional
        The safe port range for SageMaker, by default None.
    metrics_port : int, optional
        The metrics port to use, by default 8002.
    allow_http : bool, optional
        Allow HTTP, by default True.
    allow_grpc : bool, optional
        Allow gRPC, by default True.
    allow_metrics : bool, optional
        Allow metrics, by default True.
    allow_sagemaker : bool, optional
        Allow SageMaker, by default True.
    grpc_root_cert : Optional[str], optional
        The gRPC root certificate, by default None.
    grpc_server_cert : Optional[str], optional
        The gRPC server certificate, by default None.
    grpc_server_key : Optional[str], optional
        The gRPC server key, by default None.
    grpc_use_ssl : bool, optional
        Use gRPC SSL, by default False.
    grpc_use_ssl_mutual : bool, optional
        Use gRPC mutual SSL, by default False.
    log_verbose : Optional[bool], optional
        Verbose logging, by default None.
    max_batch_size : int, optional
        The maximum batch size, by default 128.
    max_queue_delay_microseconds : int, optional
        The maximum queue delay in microseconds, by default 100_000.
    preserve_ordering : bool, optional
        Preserve batch ordering, by default False.
    default_timeout_microseconds : int, optional
        The default batch timeout in microseconds, by default 1_000_000.
    """
    triton_config = TritonConfig(
        http_port=kwargs.get("http_port", 8000),
        grpc_port=kwargs.get("grpc_port", 8001),
        sagemaker_port=kwargs.get("sagemaker_port", 8080),
        sagemaker_safe_port_range=kwargs.get("sagemaker_safe_port_range", None),
        metrics_port=kwargs.get("metrics_port", 8002),
        allow_http=kwargs.get("allow_http", True),
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
        max_batch_size=kwargs.get("max_batch_size", 128),
        batcher=DynamicBatcher(
            max_queue_delay_microseconds=kwargs.get("max_queue_delay_microseconds", 100_000),  # 100 milliseconds
            preserve_ordering=kwargs.get("preserve_ordering", False),
            default_queue_policy=QueuePolicy(
                timeout_action=TimeoutAction.REJECT,
                default_timeout_microseconds=kwargs.get("default_timeout_microseconds", 1_000_000),  # 1 second
                allow_timeout_override=kwargs.get("allow_timeout_override", True),
                max_queue_size=kwargs.get("max_queue_size", 1_000),
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
