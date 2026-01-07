"""
SpikeEvent - 스파이크 이벤트 계약 객체

[V2 목표]
- 뉴런을 "몰라도" 스파이크를 처리할 수 있게
- 네트워크 = 이벤트 라우터로 전환
"""

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class SpikeEvent:
    """
    뉴런 스파이크 이벤트 계약
    
    Attributes:
        t: float              # 발생 시각 [ms]
        strength: float       # 스파이크 강도 [mV]
        metabolic_cost: float # 대사 비용 [ATP]
        plasticity_signal: float  # 가소성 신호 (PTP R)
        source_id: str       # 발생 뉴런 ID
    """
    t: float
    strength: float
    metabolic_cost: float
    plasticity_signal: float
    source_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """외부 시스템 전달용 직렬화"""
        return {
            "t": self.t,
            "strength": self.strength,
            "metabolic_cost": self.metabolic_cost,
            "plasticity_signal": self.plasticity_signal,
            "source_id": self.source_id
        }
    
    def to_json(self) -> str:
        """JSON 직렬화"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpikeEvent':
        """딕셔너리에서 생성"""
        return cls(
            t=data["t"],
            strength=data["strength"],
            metabolic_cost=data["metabolic_cost"],
            plasticity_signal=data["plasticity_signal"],
            source_id=data["source_id"]
        )

