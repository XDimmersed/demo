from demo.inference.backend_base import InferenceBackend
from demo.inference.cpu_backend import CPUBackend
from demo.inference.dummy_ascend_backend import DummyAscendBackend


def create_backend(config) -> InferenceBackend:
    backend = config.backend.lower()
    if backend == "cpu":
        return CPUBackend(config)
    if backend == "ascend":
        return DummyAscendBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")

__all__ = ["create_backend", "InferenceBackend", "CPUBackend", "DummyAscendBackend"]
