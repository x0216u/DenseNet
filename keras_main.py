from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# 设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = 36
im_width = 128
batch_size = 256
epochs = 10
class_num = 2

# 创建保存模型的文件夹
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

image_path = "./dataset/"  # 猫狗数据集路径
train_dir = image_path + "training_set"  # 训练集路径
# validation_dir = image_path + "test_set"  # 验证集路径

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator(validation_split=0.2)

# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 从训练集路径读取图片
                                                           batch_size=batch_size,  # 一次训练所选取的样本数
                                                           shuffle=True,  # 打乱标签
                                                           target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                           class_mode='categorical',
                                                           subset='training')  # one-hot编码

# 训练集样本数
total_train = train_data_gen.n

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 从验证集路径读取图片
                                                         batch_size=batch_size,  # 一次训练所选取的样本数
                                                         shuffle=True,  # 打乱标签
                                                         target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                         class_mode='categorical',
                                                         subset='validation')  # one-hot编码
# 验证集样本数
total_val = val_data_gen.n


class DenseNet(object):
    def neural(self, channel, height, width, classes):
        input_shape = (height, width, channel)
        # 使用tf.keras.applications中的DenseNet121网络，并且使用官方的预训练模型
        covn_base = tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=input_shape)
        # covn_base.trainable = True

        # 冻结前面的层，训练最后5层
        # for layers in covn_base.layers[:-5]:
        #     layers.trainable = False

        # 构建模型
        model = tf.keras.Sequential()
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
        model.add(tf.keras.layers.Dense(512, activation='relu'))  # 添加全连接层
        model.add(tf.keras.layers.Dropout(rate=0.5))  # 添加Dropout层，防止过拟合
        model.add(tf.keras.layers.Dense(classes, activation='softmax'))  # 添加输出层(2分类)
        model.summary()  # 打印每层参数信息
        return model


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r-')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r-')
    plt.plot(epochs, val_loss, 'b-')
    plt.title('Training and validation loss')
    plt.show()


def train(model):
    # 编译模型
    # ope_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ope_adam = tf.keras.optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
    model.compile(optimizer=ope_adam,  # 使用adam优化器，学习率为0.0001
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 交叉熵损失函数 binary_crossentropy
                  metrics=["accuracy"])  # 评价函数

    # 回调函数1:学习率衰减
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # 需要监视的值
        factor=0.1,  # 学习率衰减为原来的1/10
        patience=2,  # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
        mode='auto',  # 当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
        verbose=1  # 如果为True，则为每次更新输出一条消息，默认值:False
    )
    # 回调函数2:保存最优模型
    checkpoint = ModelCheckpoint(
        filepath='./save_weights/myDenseNet121.ckpt',  # 保存模型的路径
        monitor='val_acc',  # 需要监视的值
        save_weights_only=False,  # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        save_best_only=True,  # 当设置为True时，监测值有改进时才会保存当前的模型
        mode='auto',  # 当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
        period=1  # CheckPoint之间的间隔的epoch数
    )
    # 开始训练
    _history = model.fit(x=train_data_gen,  # 输入训练集
                         steps_per_epoch=total_train // batch_size,  # 一个epoch包含的训练步数
                         epochs=epochs,  # 训练模型迭代次数
                         validation_data=val_data_gen,  # 输入验证集
                         validation_steps=total_val // batch_size,  # 一个epoch包含的训练步数
                         callbacks=[checkpoint])  # 执行回调函数
    # 保存训练好的模型权重
    model.save_weights('./save_weights/myNASNetMobile.ckpt', save_format='tf')
    # 拟合，具体fit_generator请查阅其他文档,steps_per_epoch是每次迭代，需要迭代多少个batch_size，validation_data为test数据，直接做验证，不参与训练
    plot_training(_history)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r-')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r-')
    plt.plot(epochs, val_loss, 'b-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    model = DenseNet().neural(channel=3, height=im_height,
                              width=im_width, classes=class_num)  # 网络

    train(model)  # 训练
