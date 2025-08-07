일반적으로 코드러너를 실행시키면 가상환경이 적용되지 않아 라이브러리가 없다고 에러가 나오며 동작하지 않는다.

Vscode에서 가상환경을 적용해서 code runner을 실행 시켜야한다.

<br>

가상환경안에서 파이썬을 실행시켜주는 파일의 위치를 찾아야한다.

```python
# team2_env 환경을 활성화한다.
conda activate team2_env

# 활성화된 환경에서 파이썬 실행 파일의 위치를 확인한다.
which python
# 출력 -> /opt/miniconda/envs/team2_env/bin/python
```

이 경로가 **code runner**나 **VS Code**의 인터프리터 설정에서 사용해야 할 올바른 경로이다.

<br>

이제 **VS Code**의 `settings.json` 파일을 다시 확인하고, `code-runner.executorMap`에 설정된 경로를 위에서 찾은 **정확한 경로**로 수정한다.

```python
{
    // ... 다른 설정들 ...
    "code-runner.executorMap": {
        "python": "/opt/miniconda/envs/team2_env/bin/python -u"
        // 위에서 찾은 실제 경로로 수정한다.
    }
}
```

`-u`는 **파이썬** 실행 시 출력 버퍼링을 비활성화하는 옵션이다. 이 옵션을 사용하면 프로그램이 출력하는 내용을 즉시 터미널에 보여준다.

- 출력 버퍼링: 프로그램 출력을 바로 출력하지 않고 임시 공간인 버퍼에 저장한다. 이후 버퍼가 가득차거나 특정 조건을 만족하면 출력한다.

<br>

**참고**

<aside>

위의 방법은 `code runner`를 실행할 때만 적용된다.

만약 사용자가 `Ctrl+Shift+P`를 눌러 `Python: Select Interpreter`에서 `base` 환경을 선택한 상태라면, **VS Code**의 터미널에서 `python my_script.py`를 실행할 경우 `base` 환경의 **파이썬**이 사용된다.

**VS Code**의 인터프리터 선택 기능은 **code runner**와 별개로 작동한다.

`Ctrl+Shift+P`를 눌러 **`Python: Select Interpreter`** 명령어를 사용해 `team2_env`를 직접 선택해야한다.

</aside>
