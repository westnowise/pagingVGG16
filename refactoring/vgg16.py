from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_vgg16_model(num_classes):
    # VGG16 모델 불러오기
    vgg16 = VGG16(input_shape=(224, 224, 3))
    
    # VGG16의 레이어 리스트 가져오기
    vgg16_layers_list = vgg16.layers
    
    # 새로운 Sequential 모델 생성
    vgg16_model = Sequential() 
    
    # VGG16 모델의 레이어를 새로운 모델에 추가
    for i in range(len(vgg16_layers_list)-1):
        vgg16_model.add(vgg16_layers_list[i])
    
    # 예측에 맞게 모델 재정의 -> num_classes 개의 뉴런을 가진 Dense Layer 생성
    vgg16_model.add(Dense(num_classes, activation='softmax'))
    
    # VGG16의 일부 계층 가중치를 고정
    for layer in vgg16_model.layers[:-2]:
        layer.trainable = False

    for layer in vgg16_model.layers[2:]:
        layer.trainable = True
    
    return vgg16_model