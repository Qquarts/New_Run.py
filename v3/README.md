# 🧠 V3: Metabolic Neuron Simulation Engine

**생리학적 뉴런 시뮬레이션 파이프라인 (V3 계약 버전)**

---

## 📌 개요

V3는 **생리학적으로 정확한 뉴런 시뮬레이션 엔진**으로, 다음과 같은 핵심 특징을 가집니다:

- ✅ **V3 계약 원칙 준수**: 단일 방향화, 값 복사, ATP/S 정규화
- ✅ **완전한 대사 파이프라인**: DTG → Mito → HH Soma → Axon → Ca → PTP → Terminal
- ✅ **생리학적 정확도**: Hodgkin-Huxley 방정식, Nernst 전위, ATP 의존성
- ✅ **모듈화 설계**: 각 컴포넌트가 독립적으로 작동하며 통합 가능

---

## 🎯 V3 계약 원칙

V3는 다음 **계약 원칙**을 엄격히 준수합니다:

### **1. 단일 방향화 (Unidirectional Flow)**
- **Soma → Axon**: 신호는 소마에서 축삭으로만 전달
- **역방향 참조 금지**: `I_back` 같은 역방향 전류 제거
- **효과**: 네트워크 연결 시 순환 참조 방지

### **2. 값 복사 (Value Copying)**
- **참조 공유 금지**: `ionflow.V[:] = soma.V` 대신 `ionflow.set_V(soma.V)` 사용
- **효과**: 모듈 간 독립성 보장, 부작용 방지

### **3. 정규화 범위**
- **ATP**: `[0, 100]` 범위로 통일 (정규화된 단위)
- **S (Ca 정규화)**: `[0, 1]` 범위
- **시간 단위**: `[ms]` (밀리초)

### **4. 이벤트 기반 전달**
- **SpikeEvent**: 스파이크 이벤트를 객체로 전달
- **값 복사**: 이벤트 데이터는 값 복사로 전달

---

## 📁 파일 구조

```
v3-release/
├── README.md                    # 이 파일
├── LICENSE                      # 라이선스 (추가 필요)
├── requirements.txt             # 의존성 (추가 필요)
│
├── 00.bioneuron_config.py      # 설정 파일 (CONFIG)
├── 01.dtg_system.py            # DTG 시스템 (에너지-위상 동기화)
├── 02.mitochon_atp.py          # 미토콘드리아 (ATP 생성/소비)
├── 03.hh_soma.py               # Hodgkin-Huxley 소마
├── 04.ion_flow.py              # 이온 흐름 동역학
├── 05.myelin_axon.py           # 수초화 축삭 (Saltatory Conduction)
├── 06.ca_vesicle.py            # Ca²⁺ 소포 동역학
├── 07.ptp.py                   # PTP 가소성 (Post-Tetanic Potentiation)
├── 08.metabolic_feedback.py    # 대사 피드백
├── 09.synaptic_resonance.py    # 시냅스 공명
├── 10.terminal_release.py      # 터미널 방출
├── 11.bio_neurons_run.py       # 통합 파이프라인 (메인 실행 파일)
│
├── contracts/                   # V3 계약 객체
│   ├── __init__.py
│   ├── spike_event.py          # SpikeEvent (스파이크 이벤트)
│   ├── neuron_state.py         # NeuronState (뉴런 상태)
│   └── energy_state.py         # EnergyState (에너지 상태)
│
└── examples/                    # 사용 예시 (추가 예정)
    └── (예시 파일들)
```

---

## 🚀 빠른 시작

### **1. 설치**

```bash
# 의존성 설치
pip install numpy matplotlib pandas

# (선택) 컬러 출력을 위한 colorama
pip install colorama
```

### **2. 기본 실행**

```bash
# 통합 파이프라인 실행
python 11.bio_neurons_run.py
```

### **3. 결과**

실행 후 `logs/` 폴더에 다음 파일들이 생성됩니다:

- `table1.csv`: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η, θ-φ)
- `table2.csv`: 전도 파라미터 (v, tailV, Heat, CO₂, spikes)
- `terminal.csv`: 시냅스 방출량 (Q, p_eff)
- `saltatory_conduction.png`: 축삭 전도 시각화

---

## 📊 핵심 컴포넌트

### **1. DTGSystem (01.dtg_system.py)**
- **역할**: 에너지-위상 동기화 시스템
- **기능**: ATP와 위상(φ)을 동기화하여 생리학적 리듬 생성
- **수식**: `dE/dt = g_sync·(ATP - E) - γ·(E - E0)`, `dφ/dt = ω0 + α·(E - E0)`

### **2. Mitochondria (02.mitochon_atp.py)**
- **역할**: ATP 생성/소비 시스템
- **기능**: Glucose + O₂ → ATP 변환, Na/K 펌프 ATP 소비
- **수식**: `dATP/dt = k_transfer·(E_buf - ATP)·η - J_use`
- **정규화**: ATP는 `[0, 100]` 범위로 클램핑

### **3. HHSoma (03.hh_soma.py)**
- **역할**: Hodgkin-Huxley 소마 모델
- **기능**: Na⁺, K⁺, Leak 채널, ATP 의존 Na/K 펌프
- **수식**: `dV/dt = (I_Na + I_K + I_L + I_ext - I_pump) / C_m`
- **특징**: Q10 효과, Heat 피로 효과, Nernst 전위 동적 계산

### **4. IonFlowDynamics (04.ion_flow.py)**
- **역할**: 다중 이온 확산/전기이동 모델
- **기능**: Na⁺, K⁺, Ca²⁺, Cl⁻ 농도 계산
- **수식**: `∂C_i/∂t = D_i∇²C_i − μ_i·z_i·F·∇V`
- **V3 계약**: `set_V()` 메서드로 값 복사 보장

### **5. MyelinatedAxon (05.myelin_axon.py)**
- **역할**: 수초화 축삭 (Saltatory Conduction)
- **기능**: 노드(Node)와 인터노드(Internode) 구분, 도약 전도
- **수식**: `∂V/∂t = D(x)∂²V/∂x² - g_L(x)(V - E_L)/C_m(x) + I_Na_node(x,t)/C_m(x)`
- **V3 계약**: Soma → Axon 단일 방향화

### **6. CaVesicle (06.ca_vesicle.py)**
- **역할**: Ca²⁺ 소포 동역학
- **기능**: 스파이크 발생 시 Ca²⁺ 유입, 소포 방출 확률 계산
- **정규화**: S (Ca 정규화)는 `[0, 1]` 범위

### **7. PTPPlasticity (07.ptp.py)**
- **역할**: Post-Tetanic Potentiation (PTP) 가소성
- **기능**: 고주파 자극 후 시냅스 강화
- **수식**: `dR/dt = α·S·(1-R) - β·R`

### **8. Terminal (10.terminal_release.py)**
- **역할**: 시냅스 터미널 방출
- **기능**: Ca²⁺, PTP, DTG 위상에 따른 방출 확률 계산

---

## 🔧 설정 (CONFIG)

모든 파라미터는 `00.bioneuron_config.py`의 `CONFIG` 딕셔너리에서 관리됩니다:

```python
CONFIG = {
    "DTG": {...},        # DTG 시스템 설정
    "MITO": {...},       # 미토콘드리아 설정
    "HH": {...},         # HH 소마 설정
    "IONFLOW": {...},    # 이온 흐름 설정
    "AXON": {...},       # 축삭 설정
    "CA": {...},         # Ca 소포 설정
    "PTP": {...},        # PTP 설정
    "TERMINAL": {...},   # 터미널 설정
    "SOLVER": {...},     # 적분 방법 설정 (Euler/RK4)
}
```

---

## 📈 시뮬레이션 결과

### **생리학 파라미터 (table1.csv)**
- `t`: 시간 [ms]
- `ATP`: ATP 농도 [0,100]
- `Vm`: 막전위 [mV]
- `phi`: DTG 위상 [rad]
- `Ca`: Ca²⁺ 농도 [μM]
- `R`: PTP 강화량
- `eta`: 미토콘드리아 효율
- `delta_phi`: 위상 편차 (θ-φ)

### **전도 파라미터 (table2.csv)**
- `t`: 시간 [ms]
- `v`: 전도 속도 [m/s]
- `tailV`: 말단 전위 [mV]
- `Heat`: 열 [a.u.]
- `CO2`: CO₂ [a.u.]
- `spikes`: 스파이크 개수
- `active`: 활성 노드 개수
- `tail_peak`: 말단 피크 도달 여부

---

## 🎓 사용 예시

### **기본 파이프라인 실행**

```python
from bio_neurons_run import run_pipeline

# 기본 설정으로 실행
run_pipeline()
```

### **커스텀 설정**

```python
from bio_neurons_run import run_pipeline
from bioneuron_config import CONFIG

# 설정 수정
CONFIG["HH"]["gNa"] = 120.0  # Na⁺ 전도도 증가
CONFIG["AXON"]["coupling"] = 50.0  # 축삭 결합 강도 증가

# 실행
run_pipeline()
```

---

## 🔬 V3 계약 검증

V3 계약 원칙을 준수하는지 확인하려면:

1. **단일 방향화**: `I_back` 같은 역방향 참조가 없는지 확인
2. **값 복사**: `ionflow.set_V()` 사용 여부 확인
3. **ATP 정규화**: ATP가 `[0, 100]` 범위인지 확인
4. **S 정규화**: S (Ca 정규화)가 `[0, 1]` 범위인지 확인

---

## 📝 참고 사항

### **계산 비용**
- **기본 모드**: Euler 적분 (빠름)
- **고정확도 모드**: RK4 적분 (`CONFIG["SOLVER"]["HH"] = "rk4"`)

### **CFL 조건**
- 축삭 전도는 CFL 조건을 자동으로 준수합니다
- `dt_cfl = 0.9 * dx² / (2*D_max)` 계산 후 서브스텝으로 분할

### **시간 단위**
- 모든 시간은 **밀리초 [ms]** 단위입니다
- 전도 속도는 **미터/초 [m/s]** 단위로 계산됩니다

---

## 🤝 기여

V3는 계약 원칙을 엄격히 준수하는 생리학적 뉴런 시뮬레이션 엔진입니다.
기여 시 다음 원칙을 준수해주세요:

1. **V3 계약 원칙 준수**: 단일 방향화, 값 복사, 정규화
2. **생리학적 정확도**: 수식과 파라미터의 생리학적 근거
3. **모듈화**: 각 컴포넌트의 독립성 유지

---

## 📄 라이선스

(라이선스 정보 추가 필요)

---

## 🔗 관련 문서

- **V3 계약 원칙**: `contracts/` 폴더 참조
- **V3 → V4 업그레이드**: (별도 문서 참조)

---

## 📧 문의

(문의 정보 추가 필요)

---

**Version**: 3.0.0 (V3 계약 버전)  
**Last Updated**: 2026-01-04

