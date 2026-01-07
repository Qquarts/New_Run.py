"""
V3: 계약 객체 기반 네트워크 (Contract-Based Network)

이 패키지는 V2 폐순환 뉴런을 확장하여
"뉴런을 몰라도" 네트워크로 연결할 수 있게 합니다.

주요 구성:
- contracts: 계약 객체 (SpikeEvent, NeuronState, EnergyState)
- bio_neuron_v3: BioNeuronV3 래퍼 클래스 (예정)
- neuron_network: NeuronNetwork 클래스 (예정)
"""

__version__ = "3.0.0-alpha"

# 계약 객체 임포트
from .contracts.spike_event import SpikeEvent
from .contracts.neuron_state import NeuronState
from .contracts.energy_state import EnergyState

__all__ = [
    "SpikeEvent",
    "NeuronState",
    "EnergyState",
]

