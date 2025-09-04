# Hugging Face local download to Cache dir

매번 허깅페이스에서 모델을 다운받으면 오래걸린다.

원하는 로컬 디렉토리에 저장해두어 캐시처럼 사용하면 다운로드 받지않아 빠르게 실행할 수 있다.

<br>

```python
cache_dir = '/content/drive/MyDrive/ai_enginner/job_search/AI/cache/'
os.environ['HF_HOME'] = cache_dir
```

HF_HOME은 허깅페이스의 최상위 경로를 뜻한다.

이곳을 내가 원하는 디렉토리로 지정하면, 그 아래에 있는 캐시 디렉토리에 저장된다.

<br>

캐시는 `$HF_HOME/hub` 에 저장된다.

따로 지정하지 않으면 기본값은 `"~/.cache/huggingface" 이다.`

<br>

원래 모델, 토크나이저를 다운로드 받으면 자동으로 캐시에 저장한다.

그 경로를 `HF_HOME`으로 지정했으니 원하는 디렉토리에 저장하게된다.

<br>

snapshot_download 함수를 사용하면 특정 버전을 전체 다운로드 받을 수 있고 이미 캐시에 있는 파일은 재사용 가능하다.

모델을 직접 다운로드 받을때 사용한다.

이전에 말한것들은 `AutoModel.from_pretrained(model_name)` 같이 모델을 사용할때 다운로드 받는것이였다.

`snapshot_download`는 직접 모델을 다운로드 받기 위한 용도로 사용하는것이다.

1. 모델/리포지토리 전체를 스냅샷 단위로 로컬 캐시에 저장
2. 이미 캐시에 있는 파일은 다운로드하지 않고 재사용
3. 특정 파일만 다운로드 가능 (`allow_patterns` 사용)

반환값: 다운로드된 스냅샷 폴더 경로

<br>

## 문제발생

이렇게 똑같이 진행을 했지만, 내가 원하는 경로에 제대로 저장되지 않았다.

반드시 Transformer 라이브러리를 import 해오기전에 `HF_HOME`을 미리 환경설정 해야한다.

라이브러리를 먼저 불러오면 캐시 경로를 <mark>**전역 상수**</mark>로 저장한다.

상수로 저장했으니 더이상 변경되지 않는다.

따라서 이후에 변경하는 코드가 있어도 적용되지 않은것이다.
<br>
잘못된 코드

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

cache_dir = "/content/drive/MyDrive/ai_enginner/job_search/AI/cache/"
os.environ['HF_HOME'] = cache_dir
```

<br>
올바른 코드

```python
import os

cache_dir = "/content/drive/MyDrive/ai_enginner/job_search/AI/cache/"
os.environ['HF_HOME'] = cache_dir

from transformers import AutoModelForCausalLM, AutoTokenizer

```

<br>

참고

- https://huggingface.co/docs/huggingface_hub/ko/package_reference/environment_variables
- https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory
