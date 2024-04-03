# Clone
```bash
git clone $url
git submodule init
git submodule update
git submodule foreach git checkout master
```
- submodule update를 통해 sub project를 받으면, sub project는 detached HEAD 상태로 어떤 branch에도 속하지 않는 상태가 된다.
- sub project에서 git branch로 확인하면,

```bash
    * (detached from 826ba02)
      master
```
- 코드를 수정하기전에 이를 작업 branch로 checkout 해 주어야 한다.
- submodule 관련 명령어는 main project root 에서 실행해야 한다.

# Pull
```bash
git pull
git submodule update --remote --merge
```

- main project pull, sub project update 순으로 진행한다.
- --merge 옵션이 없이면, sub project는 다시 detached HEAD 상태가 된다.
- main project pull 하면, update 해야하는 sub project 리스트를 확인할수 있다.
- 특정 sub project만 업데이트 할때는,

```bash
# git submodule update --remote --merge <specific path to submodule>
git submodule update --remote --merge packages/app/katana
```

## 한번에 적용
```bash
git pull --recurse-submodules --jobs 10
```

- jobs 뒤 숫자는 병행하여 가져오는 실행 수를 의미한다.

# Push
- sub project, main project 순으로 진행한다.
- masin project를 push 하기 전에 sub project의 변경 사항을 모두 remote repo로 push 해 두어야 하기 때문에, 아래의 옵션을 사용하면 편하다.

```bash
# submodule이 모두 push된 상태인지 확인하고, 확인이 되면 main project를 push
git push --recurse-submodule=check
# submodule을 모두 push하고, 성공하면 main project를 push
git push --recurse-submodule=on-demand
```
- sub project 를 수정할때 마다, commit을 한 경우 위의 옵션이 편하다.

# Add Submodule
```bash
git submodule add <url> <path>
```
- sub project가 empty 상태일때는 .gitmodules에 자동으로 등록되지 않는다.
- .gitmodules 파일에 직접 등록 해야한다.

# Remove Submodule
```bash
git submodule deinit -f $path
rm -rf .git/modules/$path
rm -rf $path
```
- .gitmodules 에서 해당 부분 삭제

# Archive
```bash
# git archive --format=<format> <branch> -o <arcFile>
git archive --format=zip master -o ./rv.zip
```
- git 관련 파일 제외하고 압축하는 기능.
-  지원하는 format 목록 보기
git archive -l
