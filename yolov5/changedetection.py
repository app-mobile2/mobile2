import os
import cv2
import pathlib
import requests
from datetime import datetime
from getpass import getpass


class ChangeDetection:
    result_prev = []
    HOST = 'https://siwoopark.pythonanywhere.com'
    title = ''
    text = ''

    def __init__(self, names, mosaic="face"):
        self.states = {}
        self.mosaic = mosaic
        if mosaic == "face":
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_detector.empty():
                raise RuntimeError("얼굴 검출 모델을 열 수 없습니다.")
        username = input("Django 아이디: ")
        password = getpass("Django 비밀번호: ")
        res = requests.post(self.HOST + '/api-token-auth/', {
            'username': username,
            'password': password,
        }, timeout=15)
        res.raise_for_status()
        self.token = res.json()['token']

    def mask_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        for x, y, width, height in faces:
            face = image[y:y + height, x:x + width]
            small = cv2.resize(face, (8, 8), interpolation=cv2.INTER_AREA)
            image[y:y + height, x:x + width] = cv2.resize(
                small, (width, height), interpolation=cv2.INTER_NEAREST
            )
        return image

    def apply_mosaic(self, image, detections, names):
        if self.mosaic == "face":
            return self.mask_faces(image)
        height, width = image.shape[:2]
        small = cv2.resize(image, (max(1, width // 24), max(1, height // 24)), interpolation=cv2.INTER_AREA)
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        for *box, confidence, cls in detections:
            if names[int(cls)] == "person":
                x1, y1, x2, y2 = [int(value) for value in box]
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        return result

    def add(self, names, detected_current, save_dir, image, source="default"):
        previous = self.states.get(source, [0 for _ in names])
        appeared = [
            names[i] for i in range(len(names))
            if previous[i] == 0 and detected_current[i] == 1
        ]
        self.states[source] = detected_current[:]
        if appeared:
            self.title = appeared[-1]
            self.text = ", ".join(appeared)
            self.send(save_dir, image)

    def send(self, save_dir, image):
        now = datetime.now()
        today = datetime.now()
        save_path = os.getcwd() / save_dir / 'detected' / str(today.year) / str(today.month) / str(today.day)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        full_path = save_path / '{0}-{1}-{2}-{3}.jpg'.format(today.hour, today.minute, today.second, today.microsecond)
        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(full_path), dst):
            raise IOError(f'사진 저장 실패: {full_path}')
        headers = {'Authorization': 'Token ' + self.token, 'Accept': 'application/json'}
        data = {
            'title': self.title,
            'text': self.text,
            'created_date': now,
            'published_date': now,
        }
        with full_path.open('rb') as photo:
            res = requests.post(self.HOST + '/api_root/Post/', data=data, files={'image': photo}, headers=headers, timeout=15)
        res.raise_for_status()
        print(res)
