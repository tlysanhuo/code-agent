"""Project-local CPU accelerator backend for slime (conversion-time only).

slime's accelerator registry only ships cuda/musa backends; on a CPU-only
conversion run (no GPU authorized) there is no fallback. This shim registers a
minimal CPU backend through slime's public register_accelerator API - vendor/
slime stays untouched. Select with SLIME_ACCELERATOR=cpu.
"""

from __future__ import annotations

import torch

from slime.utils.accelerator import register_accelerator
from slime.utils.accelerator.torch_accelerator import TorchAccelerator


class _CpuNamespace:
    """torch.cuda-like namespace with no-op semantics for CPU execution."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def set_device(index) -> None: ...

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def synchronize(device=None) -> None: ...

    @staticmethod
    def empty_cache() -> None: ...

    @staticmethod
    def manual_seed(seed: int) -> None:
        torch.manual_seed(seed)

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        torch.manual_seed(seed)

    @staticmethod
    def get_rng_state() -> torch.Tensor:
        return torch.get_rng_state()

    @staticmethod
    def set_rng_state(state: torch.Tensor) -> None:
        torch.set_rng_state(state)

    @staticmethod
    def initial_seed() -> int:
        return int(torch.initial_seed())

    # memory-stat stubs (slime's print_memory touches these on the CPU path)
    @staticmethod
    def memory_allocated(device=None) -> int:
        return 0

    @staticmethod
    def max_memory_allocated(device=None) -> int:
        return 0

    @staticmethod
    def memory_reserved(device=None) -> int:
        return 0

    @staticmethod
    def max_memory_reserved(device=None) -> int:
        return 0

    @staticmethod
    def reset_peak_memory_stats(device=None) -> None: ...

    @staticmethod
    def mem_get_info(device=None) -> tuple[int, int]:
        return (0, 0)


class CPUAccelerator(TorchAccelerator):
    name = "cpu"
    device_type = "cpu"
    communication_backend_name = "gloo"

    def _module(self):
        return _CpuNamespace

    def is_available(self) -> bool:
        return True

    def device_name(self, index=None) -> str:
        return "cpu"

    def device(self, index=None) -> torch.device:
        return torch.device("cpu")

    def set_device(self, index) -> None: ...

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def synchronize(self, device=None) -> None: ...

    def empty_cache(self) -> None: ...

    def distributed_device_id(self, index=None) -> None:
        # gloo init_process_group accepts device_id=None
        return None


register_accelerator(
    "cpu",
    CPUAccelerator,
    lambda: True,
    priority=-100,
    communication_backends=("gloo",),
)
