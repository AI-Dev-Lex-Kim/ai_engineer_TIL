# sys.path

## sys.path 기본 구조

`sys.path`는 파이썬 인터프리터가 모듈을 import할 때 뒤지는 **검색 경로 리스트이다.**

이 리스트 안의 디렉토리를 차례대로 뒤지면서 원하는 모듈을 찾는다.

<br>

---

## sys.path에 들어가는 경로의 기준

파이썬이 실행될 때 자동으로 `sys.path`에 채워 넣는 경로들은 다음과 같다.

<br>

### **실행 스크립트의 위치**

`python some_script.py`로 실행하면, 현재 터미널에 위치한 경로 디렉토리(빈 문자열 `''` → 현재 디렉토리)가 `sys.path[0]`에 들어간다.

그래서 현재 실행하는 파일이 있는 폴더 안의 모듈은 그냥 import 할 수 있게 된다.

```bash
cd /content/drive/MyDrive/ai_enginner/job_search/
python src/llm/generator.py
```

파이썬은 이 빈 문자열을 현재 디렉토리(`/content/drive/MyDrive/ai_enginner/job_search/`)로 해석해서 import 검색을 한다.

<br>

### **PYTHONPATH 환경 변수**

환경 변수 `PYTHONPATH`에 등록된 경로들이 `sys.path`에 추가된다.

예:

```bash
export PYTHONPATH=/content/drive/MyDrive/ai_enginner/job_search
```

이렇게 해두면 파이썬 실행 시 그 경로가 `sys.path`에 들어가고, 해당 경로 밑에 있는 패키지를 import할 수 있다.

<br>

### **표준 라이브러리 경로**

파이썬이 설치될 때 같이 깔린 표준 라이브러리 디렉토리들이 들어간다.

예: `/usr/lib/python3.10/`, `/usr/lib/python3.10/lib-dynload/`

<br>

### **site-packages**

pip로 설치하는 서드파티 라이브러리들이 들어 있는 경로가 자동으로 들어간다.

예: `/usr/local/lib/python3.10/dist-packages/`
