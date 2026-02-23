from pathlib import Path
import pandas as pd


def load_data(filepath):
    """
    CSV, PKL, 또는 Parquet 파일로부터 데이터를 로드합니다.
    Path 객체와 문자열 모두 지원합니다.
    """

    # 🔹 Path 객체로 통일
    filepath = Path(filepath)

    # 🔹 파일 존재 확인
    if not filepath.exists():
        # fallback: 루트/data 폴더에서 다시 탐색
        alt_path = Path(__file__).resolve().parent.parent / "data" / filepath.name
        if alt_path.exists():
            filepath = alt_path
        else:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    print(f"{filepath}에서 데이터를 로드하는 중...")

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    elif filepath.suffix == ".pkl":
        df = pd.read_pickle(filepath)
    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("지원되지 않는 형식입니다. (.csv, .pkl, .parquet)")

    print(f"데이터 로드 완료: {df.shape}")
    return df