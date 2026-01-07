# 🧠 V3: Metabolic Neuron Simulation Engine

**생리학적 뉴런 시뮬레이션 파이프라인 (V3 계약 버전)**

---

## 📌 빠른 시작

### **1. 설치**
```bash
pip install numpy matplotlib pandas
```

### **2. 실행**
```bash
python 11.bio_neurons_run.py
```

### **3. 결과**
`logs/` 폴더에 시뮬레이션 결과가 생성됩니다.

---

## 🎯 V3 계약 원칙

V3는 다음 **계약 원칙**을 엄격히 준수합니다:

- ✅ **단일 방향화**: Soma → Axon (역방향 참조 금지)
- ✅ **값 복사**: 참조 공유 금지 (`ionflow.set_V()`)
- ✅ **ATP 정규화**: `[0, 100]` 범위
- ✅ **S 정규화**: `[0, 1]` 범위
- ✅ **시간 단위**: `[ms]` (밀리초)

---

## 📁 폴더 구조

```
v3/
├── README.md                    # 이 파일 (메인 설명)
├── 11.bio_neurons_run.py       # 메인 실행 파일
├── components/                  # 핵심 부품 모듈
│   ├── 00.bioneuron_config.py  # 설정
│   ├── 01.dtg_system.py        # DTG 시스템
│   ├── 02.mitochon_atp.py      # 미토콘드리아
│   ├── 03.hh_soma.py           # HH 소마
│   ├── 04.ion_flow.py          # 이온 흐름
│   ├── 05.myelin_axon.py       # 수초화 축삭
│   ├── 06.ca_vesicle.py        # Ca 소포
│   ├── 07.ptp.py               # PTP 가소성
│   ├── 08.metabolic_feedback.py # 대사 피드백
│   ├── 09.synaptic_resonance.py # 시냅스 공명
│   └── 10.terminal_release.py  # 터미널 방출
├── contracts/                   # V3 계약 객체
│   ├── spike_event.py
│   ├── neuron_state.py
│   └── energy_state.py
├── requirements.txt            # 의존성
├── V3_CONTRACT.md              # V3 계약 원칙 상세
└── CHANGELOG.md                # 변경 이력
```

---

## 📊 핵심 컴포넌트

### **파이프라인 흐름**
```
DTGSystem → Mitochondria → HHSoma → MyelinatedAxon
    → CaVesicle → PTP → Terminal
```

### **주요 기능**
- **DTGSystem**: 에너지-위상 동기화
- **Mitochondria**: ATP 생성/소비
- **HHSoma**: Hodgkin-Huxley 소마 모델
- **MyelinatedAxon**: 수초화 축삭 전도
- **CaVesicle**: Ca²⁺ 소포 동역학
- **PTP**: Post-Tetanic Potentiation
- **Terminal**: 시냅스 방출

---

## 🔧 설정

모든 파라미터는 `components/00.bioneuron_config.py`의 `CONFIG`에서 관리됩니다.

```python
CONFIG = {
    "DTG": {...},
    "MITO": {...},
    "HH": {...},
    "AXON": {...},
    ...
}
```

---

## 📈 결과 파일

시뮬레이션 실행 후 `logs/` 폴더에 생성:

- `table1.csv`: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η)
- `table2.csv`: 전도 파라미터 (v, tailV, Heat, CO₂)
- `terminal.csv`: 시냅스 방출량
- `saltatory_conduction.png`: 축삭 전도 시각화

---

## 📚 상세 문서

- **V3_CONTRACT.md**: V3 계약 원칙 상세 설명
- **CHANGELOG.md**: 버전별 변경 이력

---

## 🚀 사용 예시

### **기본 실행**
```python
from bio_neurons_run import run_pipeline
run_pipeline()
```

### **커스텀 설정**
```python
from bio_neurons_run import run_pipeline
from components.bioneuron_config import CONFIG

CONFIG["HH"]["gNa"] = 120.0  # Na⁺ 전도도 조정
run_pipeline()
```

---

## 🤝 기여

V3 계약 원칙을 준수하여 기여해주세요:
1. 단일 방향화 원칙
2. 값 복사 원칙
3. 정규화 범위 준수

---

**Version**: 3.0.0  
**Last Updated**: 2026-01-04
