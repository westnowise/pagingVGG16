import numpy as np
from PIL import Image

def predict_image(image_path, model):
    class_names = [
        "Mop", "damage", "defective", "explode", "frame", "furniture", 
        "gap", "gypsum", "interval", "joint", "kink", "mold", "molding", 
        "perforated", "piece", "pollution", "rust", "spot", "wall"
    ]

    # 이미지 불러오기
    image = Image.open(image_path)
    image = image.resize((224, 224))  # 모델 입력 크기에 맞게 이미지 크기 조정
    image_array = np.expand_dims(np.array(image), axis=0)

    # 모델 예측
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)

    # 예측된 클래스 이름
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name