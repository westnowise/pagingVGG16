# VGG16 모델을 활용한 도배 하자 유형 분류
DACON 도배 하자의 유형 분류 AI 모델 개발

https://dacon.io/competitions/official/236082/overview/description

[배경] 
한솔데코는 끊임없는 도전을 통해 성장을 모색하고자 하는 기치를 갖고, 공동 주택 내 실내 마감재 공사를 수행하며 시트와 마루, 벽면, 도배 등 건축에서 빼놓을 수 없는 핵심적인 자재를 유통하고 있습니다.
실내 마감재는 건축물 내부 공간의 인테리어와 쾌적한 생활을 좌우하는 만큼, 제품 결함에 대한 꼼꼼한 관리 역시 매우 중요합니다.
이를 위해 한솔데코에서는 AI 기술을 활용하여 하자를 판단하고 빠르게 대처할 수 있는 혁신적인 방안을 모색하고자 합니다.
이미지 데이터를 기반으로 도배의 하자 유형을 정확하게 분류해 낼 수 있는 AI 모델을 개발하세요!


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
    main.py



## Data Preprocessing


