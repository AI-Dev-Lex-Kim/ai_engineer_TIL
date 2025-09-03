- [1. os.path.abspath(path)](#1-ospathabspathpath)
- [2. os.path.basename(path)](#2-ospathbasenamepath)
- [3. os.path.commonpath(list\_of\_paths)](#3-ospathcommonpathlist_of_paths)
- [4. os.path.commonprefix(list\_of\_paths)](#4-ospathcommonprefixlist_of_paths)
- [5. os.path.dirname(path)](#5-ospathdirnamepath)
- [6. os.path.exists(path)](#6-ospathexistspath)
- [7. os.path.expanduser(path)](#7-ospathexpanduserpath)
- [8. os.path.expandvars(path)](#8-ospathexpandvarspath)
- [9. os.path.getatime(path)](#9-ospathgetatimepath)
- [10. os.path.getmtime(path)](#10-ospathgetmtimepath)
- [11. os.path.getctime(path)](#11-ospathgetctimepath)
- [12. os.path.getsize(path)](#12-ospathgetsizepath)
- [13. os.path.isabs(path)](#13-ospathisabspath)
- [14. os.path.isfile(path)](#14-ospathisfilepath)
- [15. os.path.isdir(path)](#15-ospathisdirpath)
- [16. os.path.join(path, \*paths)](#16-ospathjoinpath-paths)
- [17. os.path.normpath(path)](#17-ospathnormpathpath)
- [18. os.path.realpath(path)](#18-ospathrealpathpath)
- [19. os.path.ALLOW\_MISSING (상수)](#19-ospathallow_missing-상수)
- [20. os.path.relpath(path, start=os.curdir, \*, allow\_missing=False)](#20-ospathrelpathpath-startoscurdir--allow_missingfalse)
- [21. os.path.samefile(path1, path2)](#21-ospathsamefilepath1-path2)
- [22. os.path.split(path)](#22-ospathsplitpath)
- [23. os.path.splitroot(path) (Python 3.12+)](#23-ospathsplitrootpath-python-312)
- [24. os.path.splitext(path)](#24-ospathsplitextpath)

파이썬 경로와 관련된 유용한 함수

### 1. os.path.abspath(path)

상대 경로를 절대 경로(시스템 루트부터 시작하는 전체 경로)로 바꿔준다.

```python
# 상대 경로
path = "test_folder/file.txt"

# 절대 경로로 변환
abs_path = os.path.abspath(path)
print(abs_path)
```

결과 예시

```
/content/test_folder/file.txt
```

가장 최상위 경로에 나의 경로가 추가된것이다.

최상위 경로가 `‘/content’`이다. 이 부분 뒤에 내가 설정한 경로가 뒤에 붙는다.

없는 경로 이름을 넣어도 `‘/content/[경로]’` 이렇게 나온다.

---

### 2. os.path.basename(path)

경로에서 마지막 요소(파일 이름이나 디렉토리 이름)만 가져온다.

```python
path = "/home/user/data/file.txt"

print(os.path.basename(path))  # file.txt
```

---

### 3. os.path.commonpath(list_of_paths)

여러 경로에서 **공통되는 디렉토리 경로**를 찾아준다.

```python
paths = ["/home/user/docs/file1.txt", "/home/user/docs/file2.txt"]
print(os.path.commonpath(paths))  # /home/user/docs

paths = ["/home/user/docs/file1.txt", "/user/docs/file2.txt"]
print(os.path.commonpath(paths))  # /
```

---

### 4. os.path.commonprefix(list_of_paths)

여러 경로에서 **문자열 기준으로 공통 접두사**를 찾아낸다. (경로 단위가 아니라 문자 단위라 주의)

```python
paths = ["/home/user/docs/file1.txt", "/home/user/downloads/file2.txt"]
print(os.path.commonprefix(paths))  # /home/user/do

paths = ["/home/user/docs/file1.txt", "/user/downloads/file2.txt"]
print(os.path.commonprefix(paths))  # /
```

---

### 5. os.path.dirname(path)

경로에서 **디렉토리 부분만** 가져온다.

```python
path = "/home/user/data/file.txt"
print(os.path.dirname(path))  # /home/user/data
```

---

### 6. os.path.exists(path)

파일이나 디렉토리가 실제로 존재하는지 확인한다.

```python
path = "/home/user/data"
print(os.path.exists(path))  # True

path = "/home/user/data/main.py"
print(os.path.exists(path))  # True
```

---

### 7. os.path.expanduser(path)

경로에 `~` 기호가 있으면, 현재 사용자의 홈 디렉토리로 바꿔준다.

```python
print(os.path.expanduser("~/data/file.txt"))
# /Users/username/data/file.txt (맥/리눅스)
```

---

### 8. os.path.expandvars(path)

경로에 환경 변수(`$HOME`, `%USERPROFILE%` 등)가 있으면 실제 값으로 치환한다.

```python
print(os.path.expandvars("$HOME/data/file.txt"))
# /Users/username/data/file.txt

```

---

### 9. os.path.getatime(path)

파일에 마지막으로 접근(access)한 시간을 타임스탬프(초)로 반환한다.

```python
path = "example.txt"
open(path, "w").close()  # 빈 파일 생성

print(os.path.getatime(path)) # 1756865384.4164214
print(time.ctime(os.path.getatime(path)))  # 사람이 읽기 좋은 형식
																					 # Wed Sep  3 02:09:44 2025
```

---

### 10. os.path.getmtime(path)

파일이 마지막으로 수정(modify)된 시간을 반환한다.

```python
path = "example.txt"
print(time.ctime(os.path.getmtime(path))) # Wed Sep  3 02:10:30 2025
```

---

### 11. os.path.getctime(path)

파일 생성(create) 시간(운영체제마다 의미가 다를 수 있음, 윈도우는 생성 시간, 유닉스는 inode 변경 시간)을 반환한다.

```python
path = "example.txt"
print(time.ctime(os.path.getctime(path))) # Wed Sep  3 02:11:33 2025
```

---

### 12. os.path.getsize(path)

파일의 크기를 바이트 단위로 반환한다.

```python
path = "example.txt"
print(os.path.getsize(path))  # 30333
```

---

### 13. os.path.isabs(path)

경로가 **절대 경로**인지 확인한다.

절대 경로는 루트(/ 또는 드라이브 문자)부터 시작하는 전체 경로를 말한다.

```python
print(os.path.isabs("/home/user/file.txt"))  # True
```

---

### 14. os.path.isfile(path)

경로가 실제 존재하는 **파일**인지 확인한다.

디렉토리거나 없는 경로면 False를 반환한다.

```python
print(os.path.isfile("test.txt"))   # True (파일임)
print(os.path.isfile("/content/test/"))   # False (디렉토리)
```

---

### 15. os.path.isdir(path)

경로가 실제 존재하는 **디렉토리**인지 확인한다.

```python
print(os.path.isdir("/content/test/"))   # True (현재 디렉토리)
print(os.path.isdir("test.txt"))         # False (파일임)
```

---

### 16. os.path.join(path, \*paths)

경로들을 안전하게 합쳐준다.

운영체제에 맞는 구분자(`/` 또는 `\`)를 사용해 경로를 만든다.

```python
print(os.path.join("/home/user", "docs", "file.txt"))
# /home/user/docs/file.txt (리눅스/맥)
```

### 17. os.path.normpath(path)

경로를 정규화한다.

`.`(현재 디렉토리), `..`(상위 디렉토리), 불필요한 `/`를 정리해 준다.

```python

print(os.path.normpath("/home/user/../docs//file.txt"))
# /home/docs/file.txt

```

---

### 18. os.path.realpath(path)

심볼릭 링크나 가상 경로를 따라가서 **실제 물리 경로**를 반환한다.

```python

open("real.txt", "w").close()
os.symlink("real.txt", "link.txt")

print(os.path.realpath("link.txt"))  # 실제 파일 real.txt의 절대경로 출력

```

---

### 19. os.path.ALLOW_MISSING (상수)

`relpath()` 같은 함수에서 경로가 존재하지 않아도 허용하도록 하는 옵션.

Python 3.12부터 추가되었다.

```python

print(os.path.relpath("not_exist/file.txt", "base", os.path.ALLOW_MISSING))
# not_exist/file.txt  (존재하지 않아도 에러 안냄)

```

---

### 20. os.path.relpath(path, start=os.curdir, \*, allow_missing=False)

기준 디렉토리(start)에서 대상 경로(path)까지의 **상대 경로**를 계산한다.

```python

print(os.path.relpath("/home/user/docs/file.txt", "/home/user"))
# docs/file.txt

```

---

### 21. os.path.samefile(path1, path2)

두 경로가 같은 파일을 가리키면 True. (실제로 같은 inode인지 확인)

```python

open("file1.txt", "w").close()
os.symlink("file1.txt", "link1.txt")

print(os.path.samefile("file1.txt", "link1.txt"))  # True

```

### 22. os.path.split(path)

경로를 (디렉토리 부분, 마지막 요소) 튜플로 나눈다.

```python

print(os.path.split("/home/user/docs/file.txt"))
# ('/home/user/docs', 'file.txt')

```

### 23. os.path.splitroot(path) (Python 3.12+)

루트(드라이브, UNC, 루트 `/`)와 나머지 경로를 분리한다.

```python

print(os.path.splitroot("C:\\Users\\Admin\\file.txt"))
# ('C:\\', 'Users\\Admin\\file.txt')

print(os.path.splitroot("/home/user/file.txt"))
# ('/', 'home/user/file.txt')

```

---

### 24. os.path.splitext(path)

파일 이름과 확장자를 분리한다.

```python

print(os.path.splitext("archive.tar.gz"))
# ('archive.tar', '.gz')

print(os.path.splitext("file.txt"))
# ('file', '.txt')

```

참고

- https://docs.python.org/ko/3.13/library/os.path.html
