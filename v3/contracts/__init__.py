"""
계약 객체 (Contract Objects)

V3의 핵심은 "계약 객체"입니다.
뉴런의 내부 구현을 몰라도 표준화된 객체를 통해
상태를 읽고 이벤트를 전달할 수 있습니다.
"""

from .spike_event import SpikeEvent
from .neuron_state import NeuronState
from .energy_state import EnergyState

__all__ = [
    "SpikeEvent",
    "NeuronState",
    "EnergyState",
]
