from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.layers import Dense,GlobalAveragePooling2D
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.callbacks import EarlyStopping
import json
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

# Создание тренеровочных данных 
train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',  
    subset='training'
)

# Создание тестовых данных 
val_data = datagen.flow_from_directory(
    'dataset',
    batch_size=32,
    target_size=(224,224),
    class_mode='categorical',  
    subset='validation'
)

# Инициализация MobileNetV2 модели
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

# Создание итоговой модели
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),  
    Dense(train_data.num_classes, activation='softmax') 
])

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Получение истории обучения и сохранение модели
history = model.fit(train_data,validation_data=val_data,epochs=7,callbacks=[early_stopping])
model.save('model.h5')

# Сохранение истории
with open('history.json', 'w') as f:
    json.dump(history.history, f)


# График точности
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность модели на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.show()

# График потерь
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Потери модели на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)
plt.show()
