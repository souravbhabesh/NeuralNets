import numpy as np

def imagePadding(img, filter):
    filter_size = filter.shape[0]
    # Padding the input image with 0's on either side of width filter-1
    padded_img_shape = (img.shape[0]+2*filter_size-2, img.shape[1]+2*filter_size-2)
    img_padded = np.zeros(padded_img_shape)
    img_padded[filter_size-1:-filter_size+1, filter_size-1:-filter_size+1] = img
    # print(img_padded)
    return img_padded

def featureMap(img, filter):
    # stride - 1x1
    img = imagePadding(img,filter)
    # shape of filter which is a square matrix
    filter_size = filter.shape[0]
    img_feature_extracted = np.zeros((img.shape[0] - filter_size, img.shape[1] - filter_size))
    # co-ordinates which left corner of the filter will visit
    coords = [(x, y) for x in range(img.shape[0] - filter_size) for y in range(img.shape[1] - filter_size)]
    for cord in coords:
        img_feature_extracted[cord] = np.sum(
                                np.multiply(img[cord[0]:cord[0] + filter_size, cord[1]:cord[1] + filter_size], filter))
    print("Dimension of img after applying filter {0}".format(img_feature_extracted.shape))
    return img_feature_extracted


def featureMapNormalization(img):
    if np.linalg.norm(img) > 0:
        # print(img)
        # print(img / np.linalg.norm(img))
        return img / np.linalg.norm(img)
    else:
        print("Norm of the feature extracted image is 0")
        return img

def maxPooling(img, pool_window_lenght):
    img_size = img.shape[0]
    # pooled_img = np.zeros((img_size//pool_window_lenght, img_size//pool_window_lenght))
    pooled_img = []
    coords = [(x,y) for x in range(0, img_size-1, 2) for y in range(0, img_size-1, 2)]
    for cord in coords:
        pooled_img.append(np.max(img[cord[0]:cord[0]+pool_window_lenght, cord[1]:cord[1]+pool_window_lenght]))
    pooled_img = np.asarray(pooled_img)
    pooled_img = pooled_img.reshape((img_size//pool_window_lenght, img_size//pool_window_lenght))
    print("Shape of pooled image: {0}".format(pooled_img.shape))
    print(pooled_img)
    return pooled_img


if __name__ == '__main__':
    print("Applying filter to the image")
    mu = 10
    sigma = 2.0
    # img = sigma*np.random.randn(10, 10) + mu
    # Generate image with one square in it
    img = np.zeros((10, 10), dtype=np.float)
    cords = [(x, y) for x in range(10) for y in range(10)]
    # print(cords)
    # Left corner of square in image (3,3) k=3
    k = 1
    img[k:k + 5, k:k + 5] = 1

    # Sobel filter for Horizontal edge detection
    horizontal_filter = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
    horizontal_filter = horizontal_filter.reshape((3, 3))
    vertical_filter = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
    vertical_filter = vertical_filter.reshape((3, 3))

    # filter = np.eye(3)

    # Testing padding
    # print(img.shape)
    # print(img)
    # print(imagePadding(img,filter).shape)
    # imagePadding(img,filter)

    maxPooling(featureMapNormalization(featureMap(img, vertical_filter)), 2)
    # Display the image
    from matplotlib import pyplot as plt

    plt.imshow(maxPooling(featureMapNormalization(featureMap(img, vertical_filter)), 2), interpolation='nearest', cmap='gray')
    plt.show()

    plt.imshow(maxPooling(featureMapNormalization(featureMap(img, horizontal_filter)), 2), interpolation='nearest',
               cmap='gray')
    plt.show()
