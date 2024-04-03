


# Run loopABC
```bash
DCC rez-env loopABC -- loopABC [alembic file path]
```
- 해당 alembic경로에 (filename)_ loop.abc 파일이 만들어진다.

# Files
- bin/loopABC : loopABC.py 실행
- script/loopABC.py : 후디니 파일 열어서 셋팅 후 alembic export
- source/loopABC.hip : looping 하는 기본 후디니 노드
- test file : /show/pipe/template/loopABC
```note
speedTree에서 alembic export 할 때, Group by 를 “Generator name” 으로 지정 한다.
```

# Help
loopABC [looping frame 길이] [-c] [-f fps] [-s] [-h] [abc 파일 경로]

[looping frame]
- 원하는 looping frame 길이 입력 (default : 100)
```bash
loopABC 30 ./test.abc
```

[-c]
- 플래그를 등록하면, 입력된 abc 파일의 frame 정보 확인 할 수 있습니다.
```bash
loopABC -c ./test.abc
```

[-f fps]
- fps 값 입력 (default : 24)
```bash
loopABC 50 -f 29.97 ./test.abc
```

[-s]
- 플래그 등록시, 후디니 파일 저장 (abc 파일 경로에 abc파일이름.hip 으로 저장됨)
```bash
loopABC 48 -s ./test.abc
```

[-h]
- 사용법 확인


[abc 파일 경로]
- loop 하려하는 alembic 파일 경로
