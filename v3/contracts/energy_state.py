"""
EnergyState - 에너지 상태 계약 객체

[V2 목표]
- 에너지 관리를 독립적으로 추적
- 대사 피드백 루프 자동화
"""

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class EnergyState:
    """
    에너지 상태 계약
    
    Attributes:
        ATP: float           # ATP 농도
        E_buf: float         # 에너지 버퍼
        efficiency: float    # 효율 (η)
        Heat: float          # 열
        CO2: float           # CO₂
        production_rate: float  # ATP 생산율
        consumption_rate: float # ATP 소비율
    """
    ATP: float
    E_buf: float
    efficiency: float
    Heat: float
    CO2: float
    production_rate: float
    consumption_rate: float
    
    def is_critical(self) -> bool:
        """에너지 위기 상태"""
        return self.ATP < 50.0
    
    def is_optimal(self) -> bool:
        """에너지 최적 상태"""
        return 80.0 <= self.ATP <= 120.0
    
    def to_dict(self) -> Dict[str, Any]:
        """외부 시스템 전달용 직렬화"""
        return {
            "ATP": self.ATP,
            "E_buf": self.E_buf,
            "efficiency": self.efficiency,
            "Heat": self.Heat,
            "CO2": self.CO2,
            "production_rate": self.production_rate,
            "consumption_rate": self.consumption_rate
        }
    
    def to_json(self) -> str:
        """JSON 직렬화"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyState':
        """딕셔너리에서 생성"""
        return cls(
            ATP=data["ATP"],
            E_buf=data["E_buf"],
            efficiency=data["efficiency"],
            Heat=data["Heat"],
            CO2=data["CO2"],
            production_rate=data["production_rate"],
            consumption_rate=data["consumption_rate"]
        )

