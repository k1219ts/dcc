# RV

## 기본 실행

```bash
# /bin
./rv
```

## 패키지 묶기

- Centos-8.5
- RV-7.9.2
- Packages 에 *.rvpkg 생성

### 명령어

```bash
# /bin
./make_pkg.sh
```

## OS 배포

rv_plugins.zip 을 다운로드 후 압축을 풀어 각 OS에 맞는 실행파일 `install.(bat, command)` 를 실행

- win: `install.bat`
- mac: `install.command`

### 제작 방법

**패키지 묶기** 를 실행하여 _Packages_ 에 생성된 각 _*.rvpkg_ 를 `crossFlatform/sources/rvpkg` 로 복사 후 실행파일을 알맞게 수정

### 명령어

```bash
# /bin
./cross_platform.sh
```
