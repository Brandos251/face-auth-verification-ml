import tensorflow as tf
import os


base_path = r"C:\datasets\faceauth"

train_dir = os.path.join(base_path, "Train")
val_dir   = os.path.join(base_path, "Validation")
test_dir  = os.path.join(base_path, "Test")

# Настройки
batch_size = 4
image_size = (224, 224)  # Размер для MobileNetV2

# Функция загрузки и предобработки одного изображения
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Функция получения метки из пути
def get_label(path):
    parts = tf.strings.split(path, os.path.sep)
    label_str = parts[-2]  # 'Real' или 'Fake'
    label = tf.cast(label_str == 'Real', tf.int32)
    return label

# Комбинированная функция
def process_path(path):
    label = get_label(path)
    image = load_and_preprocess_image(path)
    return image, label

# Создание tf.data.Dataset
train_ds = tf.data.Dataset.list_files(os.path.join(train_dir, '*/*.jpg'), shuffle=True)
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.list_files(os.path.join(val_dir, '*/*.jpg'), shuffle=False)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.list_files(os.path.join(test_dir, '*/*.jpg'), shuffle=False)
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Проверка
for images, labels in train_ds.take(1):
    print(f'Форма изображений: {images.shape}')
    print(f'Метки: {labels.numpy()}')

