import numpy as np
import os
import tensorflow as tf
import h5py  # 用于保存到 HDF5 文件

# 设置路径
root_path = os.path.abspath(os.path.dirname(__file__))

# 超参数设置
learning_rate = 0.0005
epochs = 100
batch_size = 32

def load_training_data():
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = tf.one_hot(train_dataset['label'], depth=5)
    # 标准化数据，加入小常數以避免除零错误
    train_data = (train_data - np.mean(train_data, axis=0)) / (np.std(train_data, axis=0) + 1e-10)
    return train_data, train_label

def load_validation_data():
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = tf.one_hot(valid_dataset['label'], depth=5)
    # 标准化数据，加入小常數以避免除零错误
    valid_data = (valid_data - np.mean(valid_data, axis=0)) / (np.std(valid_data, axis=0) + 1e-10)
    return valid_data, valid_label

# 使用 MirroredStrategy 进行 Data Parallelism
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # 自定义模型类
    class CustomModel(tf.Module):
        def __init__(self, input_shape):
            super().__init__()
            # 使用 Xavier 初始化并增加层数
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
            x = tf.nn.dropout(x, rate=0.5)  # 增加 Dropout
            x = tf.nn.relu(tf.matmul(x, self.W2) + self.b2)
            x = tf.nn.dropout(x, rate=0.5)
            x = tf.nn.relu(tf.matmul(x, self.W3) + self.b3)
            x = tf.nn.dropout(x, rate=0.5)
            x = tf.nn.relu(tf.matmul(x, self.W4) + self.b4)
            return tf.nn.softmax(tf.matmul(x, self.W5) + self.b5)

    # 加载数据
    train_data, train_label = load_training_data()
    model = CustomModel(input_shape=train_data.shape[1])

    # 自定义损失函数，避免 log(0) 的问题
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-10):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred))

    # 自定义训练步骤
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch)
            loss = categorical_crossentropy(y_batch, predictions)
        gradients = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3, model.W4, model.b4, model.W5, model.b5])
        optimizer.apply_gradients(zip(gradients, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3, model.W4, model.b4, model.W5, model.b5]))
        return loss

    optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def distributed_train_step(x_batch, y_batch):
        per_replica_losses = strategy.run(train_step, args=(x_batch, y_batch))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# 训练模型
def train_model():
    for epoch in range(epochs):
        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(10000).batch(batch_size)
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)

        for x_batch, y_batch in distributed_dataset:
            loss = distributed_train_step(x_batch, y_batch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

    # 保存模型到 HDF5 文件
    model_save_path = os.path.join(root_path, 'YOURMODEL.h5')
    with h5py.File(model_save_path, 'w') as f:
        f.create_dataset('W1', data=model.W1.numpy())
        f.create_dataset('b1', data=model.b1.numpy())
        f.create_dataset('W2', data=model.W2.numpy())
        f.create_dataset('b2', data=model.b2.numpy())
        f.create_dataset('W3', data=model.W3.numpy())
        f.create_dataset('b3', data=model.b3.numpy())
        f.create_dataset('W4', data=model.W4.numpy())
        f.create_dataset('b4', data=model.b4.numpy())
        f.create_dataset('W5', data=model.W5.numpy())
        f.create_dataset('b5', data=model.b5.numpy())
    print(f"Model saved at {model_save_path}")

# 验证模型
def evaluate_model():
    valid_data, valid_label = load_validation_data()
    predictions = model(valid_data)
    predicted_labels = tf.argmax(predictions, axis=1)
    true_labels = tf.argmax(valid_label, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, true_labels), tf.float32))
    print(f'Accuracy: {accuracy.numpy() * 100:.2f}%')
    return accuracy.numpy()

if __name__ == "__main__":
    train_model()
    accuracy = evaluate_model()
    bonus = int(accuracy * 100 // 5)
    print(f"Bonus points based on accuracy: {bonus}")
