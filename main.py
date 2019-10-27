import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
# print(len(training_data))
# print(training_data[0][0].shape)

# x, y = training_data[0]
# print("Training data shape")
# print(x.shape)
# print(y.shape)
# Display the image
# from matplotlib import pyplot as plt
# plt.imshow(training_data[1000][0].reshape((28,28)), interpolation='nearest',cmap='gray')
# plt.show()


import dnn

net = dnn.DNN([784, 30, 10])
# print(net.feedForward(training_data[1000][0]))
net.sgd(training_data=training_data, epochs=30, mini_batch_size=10, eta=10.0, test_data=validation_data)
