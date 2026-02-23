import subprocess
import os
import sys
from pathlib import Path

def run_app():
    # 1. 현재 파일(app.py)이 위치한 루트 경로를 잡습니다
    root_path = Path(__file__).resolve().parent
    
    # 2. 실행할 대상인 main.py의 절대 경로를 설정합니다
    target_path = root_path / "app" / "main.py"
    
    # 3. 환경 변수(PYTHONPATH)에 루트를 등록하여 app/ 내부의 임포트 에러를 방지합니다
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_path) + os.pathsep + env.get("PYTHONPATH", "")
    
    # 4. 스트림릿 명령어 구성
    # --server.headless=true 옵션은 브라우저 자동 실행 여부에 따라 조절 가능합니다.
    cmd = ["streamlit", "run", str(target_path)]
    
    try:
        # subprocess만 실행하고, 파이썬 파일 내에서 직접 함수를 호출하지 않습니다
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nKeepTune 서비스를 종료합니다.")
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_app()