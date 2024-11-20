import tensorflow as tf
import numpy as np
import os
import h5py  # 確保導入 h5py 用於處理 .h5 文件

# 定義模型類，匹配訓練時的結構
class CustomModel(tf.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([input_shape, 512], stddev=np.sqrt(2.0 / input_shape)), name="W1")
        self.b1 = tf.Variable(tf.zeros([512]), name="b1")
        self.W2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=np.sqrt(2.0 / 512)), name="W2")
        self.b2 = tf.Variable(tf.zeros([256]), name="b2")
        self.W3 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=np.sqrt(2.0 / 256)), name="W3")
        self.b3 = tf.Variable(tf.zeros([128]), name="b3")
        self.W4 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=np.sqrt(2.0 / 128)), name="W4")
        self.b4 = tf.Variable(tf.zeros([64]), name="b4")
        self.W5 = tf.Variable(tf.random.truncated_normal([64, 5], stddev=np.sqrt(2.0 / 64)), name="W5")
        self.b5 = tf.Variable(tf.zeros([5]), name="b5")

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.W2) + self.b2)
        x = tf.nn.relu(tf.matmul(x, self.W3) + self.b3)
        x = tf.nn.relu(tf.matmul(x, self.W4) + self.b4)
        return tf.nn.softmax(tf.matmul(x, self.W5) + self.b5)

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(root_path, "dataset")
validation_file_path = os.path.join(dataset_path, 'validation.npz')
model_path = os.path.join(root_path, 'YOURMODEL.h5')

# 驗證文件是否存在
if not os.path.exists(validation_file_path):
    raise FileNotFoundError(f"找不到驗證文件：{validation_file_path}")

# 加載驗證數據
validation_data = np.load(validation_file_path)
validation_features = validation_data['data']
validation_labels = validation_data['label']

# 確保標籤是 one-hot 格式
num_classes = 5  # 根據你的分類數設置
if len(validation_labels.shape) == 1:  # 如果標籤是一維的，需要進行 one-hot 編碼
    validation_labels = tf.one_hot(validation_labels, depth=num_classes)

# 正規化驗證數據
validation_features = (validation_features - np.mean(validation_features, axis=0)) / (np.std(validation_features, axis=0) + 1e-10)

# 定義模型並加載權重
try:
    model = CustomModel(input_shape=validation_features.shape[1])
    with h5py.File(model_path, "r") as f:
        model.W1.assign(f["W1"][:])
        model.b1.assign(f["b1"][:])
        model.W2.assign(f["W2"][:])
        model.b2.assign(f["b2"][:])
        model.W3.assign(f["W3"][:])
        model.b3.assign(f["b3"][:])
        model.W4.assign(f["W4"][:])
        model.b4.assign(f["b4"][:])
        model.W5.assign(f["W5"][:])
        model.b5.assign(f["b5"][:])
    print(f"模型權重已成功從 {model_path} 加載")
except Exception as e:
    raise RuntimeError(f"加載權重失敗：{e}")

# 創建驗證數據集
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
validation_dataset = validation_dataset.batch(64)

# 評估模型
def evaluate_model():
    correct_predictions = 0
    total_samples = 0
    for features, labels in validation_dataset:
        predictions = model(features)
        predicted_labels = tf.argmax(predictions, axis=1)
        true_labels = tf.argmax(labels, axis=1)  # 確保 labels 是 one-hot 格式
        correct_predictions += tf.reduce_sum(tf.cast(predicted_labels == true_labels, tf.float32)).numpy()
        total_samples += labels.shape[0]
    accuracy = correct_predictions / total_samples
    print(f"驗證準確率：{accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate_model()
    bonus = int(accuracy * 100 // 5)
    print(f"基於準確率的加分：{bonus}")
