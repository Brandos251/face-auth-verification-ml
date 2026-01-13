import tensorflow as tf
import os
import time
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from preprocessing import train_ds, val_ds, test_ds

checkpoint_path = "mobilenet_checkpoint.h5"
total_epochs = 10
initial_epoch = 0

# Загрузка или создание модели
if os.path.exists(checkpoint_path):
    print("Загружаю сохранённую модель...")
    model = load_model(checkpoint_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    initial_epoch = int(model.optimizer.iterations.numpy() // len(train_ds))
    print(f"Продолжим с эпохи {initial_epoch + 1}")
else:
    print("Создаю новую модель MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Обучение с автосейвом при ошибках
try:
    print("Начинаю обучение...")
    start_time = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch
    )
    duration = time.time() - start_time
    print(f"Обучение завершено за {duration/60:.2f} минут")

except (KeyboardInterrupt, Exception) as e:
    print(f"\nОбнаружено прерывание или ошибка: {type(e).__name__}")
    print("Сохраняю модель...")
    model.save(checkpoint_path)
    print(f"Модель сохранена в: {checkpoint_path}")
    raise

# Сохраняем финальную версию
model.save("mobilenet_final.h5")
print("Финальная модель сохранена как mobilenet_final.h5")

# Оцениваем модель
loss, accuracy = model.evaluate(test_ds)
print(f"Точность на тесте: {accuracy:.2f}")