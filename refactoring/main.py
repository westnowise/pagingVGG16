from preprocessing import load_and_preprocess_data
import tensorflow as tf
from vgg16 import create_vgg16_model
from visual import plot_training_history
from test_img import predict_image

tf.keras.backend.clear_session()

train_ds, val_ds, numberOfClass = load_and_preprocess_data()

# 모델 생성
vgg16Model_2 = create_vgg16_model(num_classes=numberOfClass)
vgg16Model_2.summary()
# 모델 컴파일 및 학습
vgg16Model_2.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])

batch_size = 32
history = vgg16Model_2.fit(train_ds, 
                           validation_data=val_ds,
                           epochs=1,
                           callbacks=[
                               tf.keras.callbacks.EarlyStopping(
                                   monitor="val_loss",
                                   min_delta=1e-2,
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True
                               )
                           ]
                           )

# 훈련 히스토리 시각화
plot_training_history(history)

# 이미지 테스트
model = vgg16Model_2
# 테스트 이미지 경로
image_path = './test/006.png'

# 이미지 예측
# predicted_class_name = predict_image(image_path, model)
# print("Predicted class:", predicted_class_name)