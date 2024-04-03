# 버전
* 1.1
- 마야 레퍼런스 low 전환 기능 추가: 하단 좌측 풀다운 메뉴에서 mb(low)를 선택하면 마야 애니 파일 패키징 시 레퍼런스들을 low로 전환하는 기능
- Usd Info 확인 기능 추가: 목록의 우클릭 메뉴에 나타나는 Info...Usd를 클릭하면 해당 Usd를 Usd Manager로 여는 기능

* 1.1.3
- mb 샷 파일을 abc(알렘빅)으로 변환하여 패키징하는 기능이 추가되었습니다.
  동봉되는 XXX-frame.mb 파일은 해당 샷의 프레임 정보만 들어가 있는 비어있는 씬 파일입니다. 같은 폴더의 json 파일에도 프레임 정보가 저장됩니다.

* 1.1.4
- abc 뽑는 과정에서 생기던 오류를 수정했고 카메라 파일 패키징 안되던 문제가 수정됐습니다.

* 1.1.5
- _로 시작하는 애셋들도 목록에 나타나도록 수정되었습니다.
- 업체마다 첫 패키징 시 _global도 함께 패키징 되어야합니다.

* 1.1.6
- 이름에 ani 없는 파일도 패키징이 되도록 수정했습니다.
  mb 샷 패키징 시 파일 이름에 ani가 붙어있지 않은 경우 패키징이 안되는 현상이 있었습니다.
    
* 1.1.7
- 마야 및 알렘빅 패키징 시 마스터 usd 파일을 생성한 씬 파일을 우선적으로 패키징하도록 기능이 추가되었습니다.

* 1.1.8
- 군중 패키징 선택 기능이 추가되었습니다.
  샷에서 crowd 체크박스 추가되었으며 애셋에서 agent 체크박스가 추가되었습니다.
  샷 패키징 시 crowd 체크 후 withAsset을 체크하시면 샷에 포함된 asset의 agent폴더도 자동으로 포함시키도록 기능이 추가되었습니다.

* 1.1.9
- 애셋 mb 패키징 기능이 추가되었습니다.
  하단 풀다운 메뉴에 usd와 mb 중 하나를 선택할 수 있습니다.
  model과 rig가 패키징 됩니다.
  rig 패키징 시 groom이 있으면 같이 패키징 됩니다.
- Env 애셋 패키징을 위해 애셋 리스트의 우클릭 메뉴에 Check Env Asset이 추가되었습니다.
  해당 메뉴 클릭 시 CHECK항목에 Env로 표기되며 포인트 인스턴싱 된 프림이 로케이터로 컨버팅이 됩니다.
- preview texture 체크박스가 에셋 메뉴에 추가되었습니다.

* 1.1.10
- 샷 패키징 시 withAsset을 선택하면 트랙터 에러가 나는 부분 수정했습니다.

* 1.1.11
- 애셋 mb 패키징 시 비어있는 트랜스폼 노드에서 인스턴스 복제 시 발생하던 오류를 무시하도록 수정했습니다.
  예> habaCityB

* 1.1.12
- 애셋 mb 패키징 프리뷰 텍스쳐를 선택해도 제대로 패키징 되지 않던 문제가 수정되었습니다.

* 1.1.13
- 애셋 mb 패키징 시 텍스쳐가 간혹 제대로 패키징되지 않던 문제가 수정됐습니다.
- 프리뷰 텍스쳐 체크박스의 이름이 너무 길어서 prevtex로 변경했습니다.
- exr 텍스쳐 패키징이 필요한 경우에는 prevtex를 해제하고 model을 선택하면 원본 exr 텍스쳐가 패키징 됩니다.

* 1.1.14
- mb 패키징 시 ReplaceTarget.py 스크립트가 패키징 폴더 최상단의 scripts에 복사되도록 했습니다.
  로케이터가 존재하는 씬은 이 스크립트를 마야에서 실행하여 인스턴스를 복원할 수 있습니다.
- USD 애셋 패키징 시 모든 버전을 다 포함하도록 변경하였습니다.

* 1.1.15
- usd 샷 패키징이 더욱 정교해졌습니다.

* 1.1.16
- 애셋 mb 패키징 시 텍스쳐 형식을 지정할 수 있도록 기능이 추가되었습니다.
- 기존에 있던 prevtex 기능은 필요성이 없는 것으로 판단되어 제거하였습니다.
  (이 기능이 필요한 경우 따로 요청 주시길 바랍니다)
- 지원되는 텍스쳐 형식
  texture: disF, mask 이름이 들어간 텍스쳐는 exr로 변환, 나머지 텍스쳐는 jpg로 변환
  jpg: 모든 텍스쳐 파일을 jpg로 변환
  exr: 모든 텍스쳐 파일을 exr로 변환
  tif: 모든 텍스쳐 파일을 tif로 변환
- 프리뷰 이미지 기능이 추가되었습니다.
  (현재 _3d / shot / usd 에서만 작동됩니다)
  자세한 설명은 이 문서를 참조 바랍니다.
  /stdrepo/CSP/jungsup.han/documents/UsdPackager/USD패키징툴-1.1.16-프리뷰이미지.pdf
- 샷 패키징 목록에서 이미 패키징 되어있는 usd 파일이 존재하는 경우 INFO 컬럼에 Exists가 표시됩니다.
  (현재는 샷 usd만 검사하며 추후 애셋과 다른 형식도 업데이트 할 예정입니다)
- 샷 abc 패키징 방식이 변경되었습니다.
  기존 방식: works의 mb 파일에서 변환하던 방식
  새 방식: usd를 flatten하여 마야로 import 해서 변환하는 방식

* 1.1.17
- 애셋 프리뷰 기능이 추가되었습니다.
  {패키징 폴더}/preview/_3d/asset에 각 씬 파일명에 따른 프리뷰 이미지 폴더에 저장됩니다.
  front, side 두 개의 이미지만 생성됩니다.
  애셋 프리뷰 이미지는 아직 UI에 표시되지 않습니다.
- mb 애셋 변환 시 마야 내 텍스쳐 경로를 패키징 경로에 맞도록 sourceimage/에서 ../sourceimage/로 변경
- 애셋 패키징 시 프리뷰 에러가 발생하면 패키징이 멈추던 문제 수정

* 1.1.18
- mb 애셋 프리뷰 기능 작동 시 일부 애셋이 너무 무거워서 마야가 꺼지는 증상으로 인해 임시로 제거하였습니다.