# STAT3011

STAT3011 course project

https://docs.qq.com/doc/DZUNNQm9hRFJtVUh4

## Data crawler
Website: https://www.artmajeur.com/zh/yi-shu-pin/hui-hua

Data1: https://share.weiyun.com/hRzMyzMR

Data2: https://share.weiyun.com/PcW2hbBI

## Config the env
Prefer Tensorflow

Use Anaconda to manage the env

## Set up frame
Ref: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh-cn

Set up a frame at least to run the data

TODO

## Optimize
ResNet34-Adadelta: lr = ?, batch = 8, scale = 224*224, classes = 3, epoch = 50, test_acc = 0.60360, test_loss = 1.04668, train_acc = 0.76027, train_loss = 0.59806
ResNet34-Adam: beta_1=0.9, beta_2=0.999, epsilon=1e-07, lr = 0.001, batch = 8, scale = 224*224, classes = 3, epoch = 50, test_acc = 0.49550, test_loss = 1.36121, train_acc = 0.64954, train_loss = 0.80464
ResNet34-Adam: beta_1=0.9, beta_2=0.999, epsilon=1e-07, lr = 0.01, batch = 8, scale = 224*224, classes = 3, epoch = 50, test_acc = 0.47748, test_loss = 1.14379, train_acc = 0.70890, train_loss = 0.68479
ResNet34-Adam(can be better with more epoch): beta_1=default, beta_2=default, epsilon=default, lr = default, batch = 8, scale = 224*224, classes = 3, epoch = 50, test_acc = 0.64865, test_loss = 1.04515, train_acc = 0.73516, train_loss = 0.63898

