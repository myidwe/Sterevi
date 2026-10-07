# 0.1.4-preview · NVIDIA GPU 자동 설치

**[처음 설치와 연결](GETTING_STARTED.md)** · [GPU 지원](GPU_SUPPORT.md) · [사용 방법](DESKTOP_USER_GUIDE.md) · [문서 목차](README.md)

RTX 20·30·40·50용 실행 경로를 추가한 PC 업데이트입니다. **설치 프로그램이 실제 GPU와 드라이버를 확인하고 필요한 라이브러리를 자동 선택**합니다. GPU를 이름으로 구분하거나 사용자가 CUDA 버전을 직접 고를 필요는 없습니다.

## 다운로드

| Windows | Quest |
|:---|:---|
| [Desktop Setup EXE](https://github.com/myidwe/Sterevi/releases/download/v0.1.4-preview/Sterevi-Desktop-Setup-0.1.4-preview.exe) | [Quest Setup EXE](https://github.com/myidwe/Sterevi/releases/download/v0.1.4-preview/Sterevi-Quest-Setup-0.1.4-preview.exe) |
| PC 앱 설치·업데이트 | Windows에서 USB로 Quest 앱 설치 |
| | [APK 다운로드](https://github.com/myidwe/Sterevi/releases/download/v0.1.4-preview/Sterevi-Quest-0.1.4-preview.apk) · 별도 설치 도구 사용 |

수동 설치 ZIP, 네이티브 대응 소스를 포함한 Source ZIP, `SHA256SUMS.txt`, 개인정보 검사·검증 보고서는 [같은 릴리스](https://github.com/myidwe/Sterevi/releases/tag/v0.1.4-preview)에서 받을 수 있습니다. 직접 다운로드하는 APK와 Quest Setup에 포함된 APK는 동일합니다. 바이너리를 재빌드하려면 GitHub의 자동 Source code ZIP 대신 **Sterevi-Source ZIP**을 사용하세요.

## 기존 사용자

PC 앱을 중지하고 종료한 뒤 **새 Desktop Setup → 기존 설치 폴더 → 업데이트** 순서로 진행합니다. 설정·모델·페어링 정보와 바로가기를 유지하며, 검사가 실패하면 기존 복원 절차를 사용합니다.

**Quest 앱은 다시 설치할 필요가 없습니다.** 이번 묶음에는 기존 서명 APK **0.1.3-preview / versionCode 4**를 그대로 넣었습니다. 다운로드 파일 이름은 0.1.4-preview지만 앱 내부 버전·패키지·서명·파일 내용은 같습니다. 처음 설치하는 사용자에게도 바로 사용할 수 있는 전체 묶음을 제공합니다. 더 오래된 공개 앱을 업데이트할 때는 먼저 삭제하지 마세요.

## 바뀐 내용

| GPU의 실제 아키텍처 | 설치 라이브러리 | 실기 검증 |
|---|---|---|
| Turing · CC 7.5 · RTX 20 | Torch 2.7.1+cu126 / torchvision 0.22.1+cu126 | RTX 2060 SUPER 8GB |
| Ampere · CC 8.6 · RTX 30 | 동일한 cu126 | 미검증 |
| Ada · CC 8.9 · RTX 40 | 동일한 cu126 | 미검증 |
| Blackwell · CC 12.0 · RTX 50 | Torch 2.7.1+cu128 / torchvision 0.22.1+cu128 | 미검증 |

설치기는 라이브러리를 받거나 기존 앱을 갱신하기 전에 GPU·드라이버 조건을 확인합니다. 조건이 맞지 않으면 원인을 표시하고 중단합니다. 설치 후에는 실제 GPU에서 CUDA 커널 7종을 실행해 기준 결과와 비교하고, 고정 깊이 모델의 CUDA graph 추론을 확인한 뒤 Qt 화면까지 검사합니다. 검사 실패를 설치 완료로 표시하지 않습니다.

기존 모델·가중치·AI 해상도·좌우 합성 수식·캡처 휠·송출 호스트·기본 화질·소리 설정은 유지합니다. 이번 변경은 GPU 호환 경로를 넓히는 작업이며 새로운 화질이나 FPS 향상을 약속하지 않습니다. CUDA Toolkit을 따로 설치하거나 드라이버를 자동으로 변경하지 않습니다. 드라이버 조건과 다중 GPU 제한은 [GPU 지원 안내](GPU_SUPPORT.md)를 확인하세요.

## 확인한 내용과 남은 검증

개발 PC의 **Windows 11 / RTX 2060 SUPER 8GB**에서 cu126·cu128 두 환경의 실제 CUDA 커널과 깊이 모델 추론을 확인했습니다. 실제 화면 캡처 → AI 깊이 → 좌우 영상 → NVENC H.264 → 파일 디코딩도 확인했습니다. 다른 GPU용 PTX 컴파일과 GPU별 설치 정책 검사는 해당 GPU에서 실행했다는 뜻이 아닙니다.

**RTX 30·40·50에서의 설치·성능·캡처·인코더·Quest 연결·장시간 사용은 미검증**입니다. cu128을 RTX 2060에서 검사한 결과를 RTX 50 실기 검증으로 취급하지 않습니다. 새 Windows에서 Python 공식 설치기까지 실행하는 최초 설치와 네트워크·권한 차이도 별도 확인이 필요합니다.

검증 상태는 릴리스의 `release-validation.json`에 기록합니다. 개인정보 검사는 최종 배포 파일과 EXE 내부 ZIP·중첩 압축 소스를 대상으로 하며, Git 이력·원격 캐시·알 수 없는 암호화된 데이터의 부재까지 보장하지 않습니다.

Quest APK는 기존 파일이므로 새 APK의 착용 검증을 주장하지 않습니다. 이전에 남아 있던 최신 공개 APK의 Quest 2 확인, 영상·음성 오차 수치와 장시간 사용 검증도 남아 있습니다. 얇은 물체·가려진 배경의 윤곽 차이, DRM·캡처 차단 콘텐츠, 모든 장면의 최소 FPS 보장에 관한 제한은 그대로입니다.

## 사용 조건

Windows x64, 지원하는 NVIDIA GPU, Quest 2 / 3, 16:9 모니터와 같은 공유기의 신뢰할 수 있는 로컬 네트워크가 필요합니다. AMD·Intel GPU와 다른 CUDA 아키텍처는 지원하지 않습니다. 여러 GPU나 내장 GPU가 함께 있는 PC는 현재 기본 CUDA GPU 0과 캡처 장치의 연결 조건도 확인해야 합니다.

Windows EXE에는 아직 코드 서명이 없어 SmartScreen이나 관리 정책에 따라 경고·차단될 수 있습니다. 설치는 일반 사용자 권한으로 진행하고, **연결 허용**에 필요한 방화벽 변경이 있을 때만 Windows 승인을 요청합니다. Quest를 처음 설치하려면 개발자 모드·USB 디버깅·ADB가 필요합니다. [설치 조건](EXE_INSTALLERS.md#권한과-windows-조건)
