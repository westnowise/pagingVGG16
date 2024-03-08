import tensorflow as tf
from glob import glob

# Test and Train path

def load_and_preprocess_data():
    img_height = 244
    img_width = 244
    train_ds = tf.keras.utils.image_dataset_from_directory(
      "./train",
      validation_split=0.2,
      subset='training',
      image_size=(224, 224),
      batch_size=32,
      seed=19,
      
      shuffle=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      "./train",
      validation_split=0.2,
      subset='validation',
      image_size=(224, 224),
      batch_size=32,
      seed=19,
      
      shuffle=True)

    # 제로 패딩을 적용한 이미지 데이터셋 생성
    train_ds = train_ds.map(
        lambda x, y: (tf.image.pad_to_bounding_box(x, 0, 0, 224, 224), y)
    )

    val_ds = val_ds.map(
        lambda x, y: (tf.image.pad_to_bounding_box(x, 0, 0, 224, 224), y)
    )

    numberOfClass = len(glob("train" + "/*"))
    print("Number Of Class: ", numberOfClass)
    
    return train_ds, val_ds, numberOfClass