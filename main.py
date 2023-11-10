import coremltools as ct
import cv2
import PIL.Image
import numpy as np

counter = 0

def start_camera() -> str:
    global counter
    img_name = ''
    camera = cv2.VideoCapture(1)
    while True:
        ret, frame = camera.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            camera.release()
            counter += 1
            img_name = f"img_{counter}.png"
            cv2.imwrite(img_name, frame)
            break
    return img_name

def load_image(path, resize_to=None):
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img

def classify_image(img):
    model = ct.models.MLModel('tool_classifier.mlmodel')
    prediction = model.predict({'image': img})
    return prediction

def main() -> None:
    fname = start_camera()
    _, img = load_image(fname)
    result = classify_image(img)
    print(result)


if __name__ == '__main__':
    main()
