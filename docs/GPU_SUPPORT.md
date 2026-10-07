# NVIDIA GPU 지원

**0.1.4-preview · 2026-10-07**

RTX 20·30·40·50 시리즈용 실행 경로를 제공합니다. 설치 프로그램이 GPU의 실제 CUDA 아키텍처와 드라이버를 확인한 뒤 알맞은 라이브러리를 선택합니다. [0.1.4-preview 릴리스](https://github.com/myidwe/Sterevi/releases/tag/v0.1.4-preview)의 Desktop Setup을 사용하세요. 이전 0.1.3-preview 설치 파일은 Turing용이며 이 변경을 포함하지 않습니다.

| GPU 아키텍처 | 주요 제품 | 설치 런타임 | 커널 컴파일 대상 | 실기 검증 |
|---|---|---|---|---|
| Turing · CC 7.5 | GeForce RTX 20 | Torch 2.7.1+cu126 / CUDA 12.6 | compute_75 | RTX 2060 SUPER 8GB |
| Ampere · CC 8.6 | GeForce RTX 30 | Torch 2.7.1+cu126 / CUDA 12.6 | compute_86 | 미검증 |
| Ada · CC 8.9 | GeForce RTX 40 | Torch 2.7.1+cu126 / CUDA 12.6 | compute_89 | 미검증 |
| Blackwell · CC 12.0 | GeForce RTX 50 · 5050~5090 | Torch 2.7.1+cu128 / CUDA 12.8 | compute_120 | 미검증 |

모델 이름으로 GPU를 판단하지 않습니다. 일부 노트북 제품처럼 이름과 아키텍처 세대가 다른 경우에도 실제 capability로 선택합니다. 같은 capability를 가진 GTX·워크스테이션 제품이 검사에 통과할 수 있지만, 개별 모델의 VRAM·NVENC·실시간 성능을 보장하지 않습니다. AMD·Intel GPU, 다른 CUDA 아키텍처는 지원 범위에 포함하지 않습니다.

## 설치와 확인

새 Desktop Setup은 GPU 검사 후 필요한 Torch·torchvision만 고정 버전으로 설치합니다. GPU나 드라이버가 맞지 않으면 기존 설치와 라이브러리 다운로드를 시작하기 전에 중단하고 원인을 표시합니다. 드라이버를 자동으로 바꾸거나 별도의 CUDA Toolkit 설치를 요구하지 않습니다. NVIDIA GPU 드라이버는 사용자가 설치해야 합니다.

설치 후에는 실제 GPU에서 톤 매핑·입력 리사이즈·윤곽 보정·좌우 합성 등 7개 CUDA 연산을 실행해 기준 결과와 비교합니다. 모델 다운로드·해시 확인 뒤에는 실제 깊이 모델의 CUDA graph 추론도 확인합니다. 이 검사가 실패하면 설치 완료로 표시하지 않습니다. 업데이트 실패 시 기존 복원 절차를 사용합니다.

기존 Quest 앱·페어링·Depth·화면·소리 설정은 그대로 사용합니다. 공통 송출 경로는 H.264 / HEVC이며, GPU 지원 확장에 AV1이나 새 모델을 묶지 않았습니다. 기본 해상도나 AI 크기도 변경하지 않았습니다. 더 좋은 GPU에서의 FPS와 화질 설정은 실측 후 판단합니다.

## 드라이버와 여러 GPU

실행 중 CUDA 소스를 PTX로 컴파일하므로 일반 CUDA 12.x 최소 드라이버 조건만으로는 부족합니다. 설치 프로그램은 CUDA 드라이버 API가 cu126에 **12060 이상**, cu128에 **12080 이상**인지 확인합니다. CUDA 12.6 GA의 대응 Windows 드라이버는 560.76, CUDA 12.8 GA는 570.65 이상입니다. RTX 50은 해당 카드 자체를 지원하는 드라이버도 필요하므로 이 숫자만 보고 오래된 드라이버를 설치하지 마세요.

현재 기본 AI 장치는 **CUDA GPU 0**입니다. 여러 NVIDIA GPU나 내장 GPU가 함께 있는 노트북에서는 화면 캡처 장치와 AI 장치의 연결이 추가 조건입니다. 기존 LUID 검사와 실패 처리를 유지하며, 검사 실패를 다른 장치로 자동 우회하지 않습니다. 해당 구성의 캡처·인코더·Quest 연결은 미검증입니다.

## 소스에서 실행

고정 Python 3.12.6 x64와 프로젝트의 uv를 사용합니다. 네이티브 입력 준비는 [빌드 안내](BUILDING.md)를 따릅니다. 다른 GPU로 옮긴 기존 설치는 새 설치 프로그램으로 업데이트하는 방법을 권장합니다.

```powershell
# RTX 20 / 30 / 40: 기존 기본 경로
uv sync --locked --extra gpu-capture

# RTX 50: CUDA 12.8 경로. 기존 cu126 기본 그룹 제외
uv sync --locked --extra gpu-capture --extra gpu-blackwell --no-group gpu-default

# 선택한 환경을 다시 동기화하지 않고 실행 / 검사
uv run --no-sync python scripts/release/verify_installed_gpu.py
uv run --no-sync python scripts/release/verify_installed_depth.py
uv run --no-sync python -m quest3d.desktop_qt
```

`uv sync`의 기본값은 기존 cu126입니다. Blackwell 환경에서 수동으로 다시 동기화할 때도 위의 Blackwell 옵션을 유지하세요. `--all-extras`로 GPU 경로를 한꺼번에 선택하지 않습니다. 두 버전의 Torch를 한 환경에 설치하지 않도록 lock에 충돌 조건을 명시했습니다.

Torch는 기본 `gpu-default` 그룹 또는 `gpu-blackwell` extra로 설치합니다. 일반 `pip install .`만으로 GPU 의존성이 준비되는 구성은 아닙니다. 사용자는 EXE, 개발자는 위의 고정 uv 명령을 사용합니다. 드라이버·CUDA와 설치되는 라이브러리의 원래 약관은 유지하며, 전체 실행 환경이 GPL이라는 뜻은 아닙니다. [구성 요소 고지](../THIRD_PARTY_NOTICES.md)

## 검증 범위

현재 PC에서 CUDA 12.6·12.8 두 환경의 커널 7종, 고정 모델의 CUDA graph 추론, 실제 GPU 화면 캡처→좌우 합성→H.264 NVENC 파일 인코딩·디코딩을 확인했습니다. 새 EXE의 별도 폴더 설치에서도 RTX 2060 SUPER 감지→cu126 자동 선택→커널·모델·Qt UI 검사→설치 완료를 확인했습니다. 기존 사용자 설치·바로가기는 변경하지 않았습니다.

GPU별 정책·설치 사전 검사와 실제 커널 컴파일, 실제 GPU 실행은 구분합니다. NVRTC가 다른 아키텍처용 PTX를 생성할 수 있다는 결과는 그 GPU에서 실행했다는 증거가 아닙니다. 현재 검증 호스트는 RTX 2060 SUPER입니다. CUDA 12.8 환경을 이 GPU에서 검사해도 RTX 50의 모델·캡처·NVENC·Quest 수신·장시간 안정성 검증을 대신하지 않습니다.

새 GPU에서는 설치 검사에 이어 실제 모니터 캡처, 2D/3D 전환, Quest 영상·소리, 최소 30분의 프레임·VRAM·누적 지연을 확인해야 합니다. 출력 해상도·AI 추론 해상도·송출 FPS·깊이 갱신률을 함께 기록하며, GPU 모델명만으로 FPS를 약속하지 않습니다.

## 공식 근거

- [NVIDIA CUDA GPU 목록](https://developer.nvidia.com/cuda/gpus): 실제 compute capability
- [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) · [고정 버전 설치](https://pytorch.org/get-started/previous-versions/#v271): Blackwell 및 Windows cu128 경로
- [CUDA 12.8 릴리스 안내](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html): Toolkit·드라이버 조건
- [CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html): 오래된 드라이버의 PTX 제한
- [NVIDIA 인코더 지원표](https://developer.nvidia.com/video-encode-decode-support-matrix): 제품별 코덱 기능
