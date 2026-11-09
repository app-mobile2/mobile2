import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def monitor(frames, model, publish):
    previous = set()
    for frame in frames:
        results = model(frame[:, :, ::-1])
        current = {model.names[int(row[-1])] for row in results.xyxy[0]}
        appeared = sorted(current - previous)
        if appeared:
            annotated = results.render()[0][:, :, ::-1]
            publish(appeared, annotated)
        previous = current


def main():
    import cv2
    import requests
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--server", default="https://siwoopark.pythonanywhere.com")
    parser.add_argument("--person-only", action="store_true")
    args = parser.parse_args()
    model = torch.hub.load(str(ROOT / "yolov5"), "yolov5s", source="local")
    if args.person_only:
        model.classes = [0]
    capture = cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    try:
        if not capture.isOpened():
            raise RuntimeError("영상 입력을 열 수 없습니다.")

        def frames():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame

        def publish(names, image):
            now = datetime.now()
            directory = ROOT / "detected" / now.strftime("%Y/%m/%d")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (now.strftime("%H%M%S_%f") + ".jpg")
            if not cv2.imwrite(str(path), image):
                raise IOError(f"사진 저장 실패: {path}")
            print(f"검출 사진 저장: {path}", flush=True)
            with path.open("rb") as photo:
                response = requests.post(
                    args.server.rstrip("/") + "/api_root/Post/",
                    data={"title": f"{names[0]} 출현", "text": "새로 감지한 객체: " + ", ".join(names)},
                    files={"image": (path.name, photo, "image/jpeg")},
                    timeout=(5, 30),
                    allow_redirects=False,
                )
            if response.status_code != 201:
                raise RuntimeError(f"게시 실패 HTTP {response.status_code}: {path}")
            print(f"게시 완료: {', '.join(names)}", flush=True)

        monitor(frames(), model, publish)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
