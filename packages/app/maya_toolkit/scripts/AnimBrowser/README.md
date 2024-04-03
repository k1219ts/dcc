[2018-01-05]
  - Tag Search 기능 구현 시작

[2017-02-06]
※ Content 구성도
  - ContentTab : 좌측 directory에서 더블클릭 하여 열었을때 그 안에 있는 내용이 보이는 부분
  - ContentItem : 그 폴더 안에 들어있는 한가지 아이템을 구성한 내용

[2017-02-02]

※ directory 구성도
 - AnimationInfo : 현재 애니메이션에 대한 정보, 선택한 프리뷰가 보이도록 해주는 내용이 담긴 폴더
 - Content : 애니메이션에 종류들이 프리뷰Mov, .anim, json파일을 묶어 놓을 컨텐츠 파일
 - Item : 애니메이션을 관리하는 폴더로 애니메이션이 무엇이 있는가, 어떤 디렉토리에 담겨있는가등을 구성함
 - Pipeline : 툴에 사용함에 있어서 Import, Export등을 관리할 내용이 담긴 폴더

※ 추가사항
 [ImportFile.py]
  - Import하는 UI Dialog를 구성해놓은 파일
  - 기본적인 UI기능들 구현 완료

 [DirItem.py]
  - 폴더 목록 구성하는 TreeWidgetItem 내용 구현

Animbrowser 프로젝트 시작 [ 2017-02-02 ]