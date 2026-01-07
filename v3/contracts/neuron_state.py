"""
NeuronState - 뉴런 전체 상태 계약 객체

[V2 목표]
- 뉴런을 "몰라도" 상태를 읽을 수 있게
- 모니터링/디버깅 자동화
"""

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class NeuronState:
    """
    뉴런 전체 상태 계약
    
    Attributes:
        Vm: float            # 막전위 [mV]
        ATP: float           # ATP 농도
        Ca: float            # Ca²⁺ 농도 [μM]
        Heat: float          # 열 [a.u.]
        CO2: float           # CO₂ [a.u.]
        phi: float           # DTG 위상 [rad]
        theta: float         # 시냅스 위상 [rad]
        PTP_R: float         # PTP 강화량
    """
    Vm: float
    ATP: float
    Ca: float
    Heat: float
    CO2: float
    phi: float
    theta: float
    PTP_R: float
    
    def is_stable(self) -> bool:
        """안정성 검증"""
        return (
            -120.0 <= self.Vm <= 80.0 and
            0.0 <= self.ATP <= 200.0 and
            0.0 <= self.Ca <= 50.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """외부 시스템 전달용 직렬화"""
        return {
            "Vm": self.Vm,
            "ATP": self.ATP,
            "Ca": self.Ca,
            "Heat": self.Heat,
            "CO2": self.CO2,
            "phi": self.phi,
            "theta": self.theta,
            "PTP_R": self.PTP_R
        }
    
    def diff(self, other: 'NeuronState') -> Dict[str, float]:
        """상태 차이 계산"""
        return {
            "dVm": self.Vm - other.Vm,
            "dATP": self.ATP - other.ATP,
            "dCa": self.Ca - other.Ca,
            "dHeat": self.Heat - other.Heat,
            "dCO2": self.CO2 - other.CO2,
            "dphi": self.phi - other.phi,
            "dtheta": self.theta - other.theta,
            "dPTP_R": self.PTP_R - other.PTP_R
        }
    
    def to_json(self) -> str:
        """JSON 직렬화"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuronState':
        """딕셔너리에서 생성"""
        return cls(
            Vm=data["Vm"],
            ATP=data["ATP"],
            Ca=data["Ca"],
            Heat=data["Heat"],
            CO2=data["CO2"],
            phi=data["phi"],
            theta=data["theta"],
            PTP_R=data["PTP_R"]
        )

