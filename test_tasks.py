
"""def win1(s):
    return s+1 >= 42 or s+2 >= 42 or s*2 >= 42

def lose1(s):
    return (not(win1(s))) and (win1(s+1) and win1(s+2) and win1(s*2))

def win2(s):
    return lose1(s+1) or lose1(s+2) or lose1(s*2)

def lose2(s):
    return win2(s+1) and win1(s+2) or win1(s+1) and win2(s+2) or win2(s*2) and win1(s+1) or win2(s+1) and win1(s*2) or win2(s*2) and win1(s+2) or win2(s+2) and win1(s*2)

for i in range(1,42):

    if lose1(i):
        print(i)

"""
"""
def f19(x,y,h):
    if (h ==2 or h == 4)  and x + y >= 41:
        return 1
    elif h == 4 and x+y < 41:
        return 0
    elif x+y >= 41 and h < 4:
        return 0
    else:
        if h % 2 == 1:
            return f19(x+1,y,h+1) or f19(x*2,y,h+1) or f19(x+2,y,h+1)
        else:
            return f19(x+1,y,h+1) and f19(x*2,y,h+1) and f19(x+2,y,h+1) and f19(x,y*2,h+1)

for x in range(1,32):
    if f19(x,8,0) == 1:
        print(x)

"""

"""from functools import lru_cache 
def moves(x): 
    return x+1, 2*x, x+2
@lru_cache (None) 
def game(x):
    if any (m>=42 for m in moves(x)): return "WIN1" 
    if all (game (m) == "WIN1" for m in moves(x)): return "LOSS1"
    if any (game(m) == "LOSS1" for m in moves(x)): return "WIN2"
    if all(game (m) == "WIN1" or game(m) == "WIN2" for m in moves(x)): return "LOSS12"
for x in range(1,41):
    if game(x) == "WIN2":
        print('Для задания 20 ответ', x)
    if game(x) == "LOSS12":
        print('Для задания 21 ответ', x)"""


"""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor

df = pd.read_csv("data.csv")

scaler = MinMaxScaler()

param_grid = {
    'n_estimators':[200,300,400],
    'learning_rate': [0.005,0.01, 0.1], 
    'max_depth': [1,2,3], 
    'min_samples_split': [1,2, 5]
}

df['gender'] = df['gender'].replace({'female':1,'male':2})
df['race/ethnicity'],_ = pd.factorize(df['race/ethnicity'])
df['parental level of education'],_ = pd.factorize(df['parental level of education'])
df['lunch'],_ = pd.factorize(df['lunch'])
df['test preparation course'],_ = pd.factorize(df['test preparation course'])
df['math score'] = scaler.fit_transform(df[['math score']])


X = df[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]
y = df['math score']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = GradientBoostingRegressor(random_state=42)

cv_model = GridSearchCV(model,param_grid,cv=3,scoring='neg_mean_squared_error')

cv_model.fit(X_train,y_train)

print("Best parameters:", cv_model.best_params_)
best_model = cv_model.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_pred,y_test)

print(f'MSE of this model is: {mse}')
print(f"MAE: {mae:.4f}")  
print(f"R²: {r2:.4f}")

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
baseline_mse = mean_squared_error(y_test, dummy.predict(X_test))
print(f"Baseline MSE: {baseline_mse:.4f}")
"""

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
history = model.fit(train_data,validation_data=val_data,epochs=7)
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