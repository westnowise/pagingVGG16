# VGG16 모델을 활용한 도배 하자 유형 분류



## 🔖 Guide
### Packages Needed
    
    from glob import glob
    import numpy as np
    from PIL import Image

    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    import matplotlib.pyplot as plt


    
### Running
    pip install tensorflow
    pip install argon



경로 전부 영어로 한 후 main.py 실행하면됩니다
