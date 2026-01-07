# 📝 변경 이력 (Changelog)

## [3.0.0] - 2026-01-04

### 🎯 V3 계약 버전 릴리즈

**핵심 변경사항:**

#### ✅ V3 계약 원칙 적용
- **단일 방향화**: Soma → Axon 단일 방향화 원칙 적용, `I_back` 제거
- **값 복사**: `ionflow.set_V()` 메서드 추가, 참조 공유 금지
- **ATP 정규화**: ATP 범위를 `[0, 100]`으로 통일
- **S 정규화**: Ca 정규화(S) 범위를 `[0, 1]`로 통일
- **시간 단위**: 모든 시간을 `[ms]` 단위로 통일

#### 📦 핵심 컴포넌트
- **DTGSystem**: 에너지-위상 동기화 시스템
- **Mitochondria**: ATP 생성/소비 시스템
- **HHSoma**: Hodgkin-Huxley 소마 모델 (Q10 효과, Heat 피로 효과)
- **IonFlowDynamics**: 다중 이온 확산/전기이동 모델
- **MyelinatedAxon**: 수초화 축삭 (Saltatory Conduction)
- **CaVesicle**: Ca²⁺ 소포 동역학
- **PTPPlasticity**: Post-Tetanic Potentiation 가소성
- **Terminal**: 시냅스 터미널 방출

#### 📋 계약 객체
- **SpikeEvent**: 스파이크 이벤트 계약 객체
- **NeuronState**: 뉴런 상태 계약 객체
- **EnergyState**: 에너지 상태 계약 객체

#### 🔧 개선사항
- **CFL 조건 자동 준수**: 축삭 전도 시 CFL 조건 자동 계산 및 서브스텝 분할
- **RK4 적분 지원**: 고정확도 모드에서 RK4 적분 방법 사용 가능
- **생리학적 정확도**: Nernst 전위 동적 계산, ATP 의존 Na/K 펌프

#### 📚 문서
- **README.md**: 전체 문서화
- **V3_CONTRACT.md**: V3 계약 원칙 상세 설명
- **requirements.txt**: 의존성 패키지 목록
- **.gitignore**: Git 제외 파일 목록

---

## [이전 버전]

### V2
- 계약 객체 기반 네트워크 구조
- 단일 뉴런 시뮬레이션

### V1
- 기본 뉴런 시뮬레이션
- 단순 파이프라인

---

**Version**: 3.0.0  
**Release Date**: 2026-01-04

