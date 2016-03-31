
from numpy import *


#wrong way?
x = 1
for i in range(0,1000000):
    x = x + 1e-6
x = x - 1
#Out[45]: 0.95367431640625

#write way?
1e9 + sum([1e-6 for i in range(0,1000000)]) - 1e9
#Out[46]: 1.0


#if you center at zero, it'll be well conditioned, and the
#optimization will be ez to do, 0 mean and unit variance

#LOSS

scores = [3.0, 1.0, 0.2]

def softmax(x):
    return exp(x) / sum(exp(x), axis=0)

print(softmax(scores))

import matplotlib.pyplot as plt
x = arange(-2.0, 6.0, 0.1)

scores = np.vstack([x, numpy.ones_like(x)])


def load_letter(folder, minnum):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),dtype=np.float32)
    image_index = 0,
    print(folder)
    for image in os.listdir(folder):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                            pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    numI= image_index
    dataset = dataset[0:, :, :]
    if numI< minnum:
        raise Exception('Many od fewer images than expected: %d < %d' %
                        (numI, minnum))
    print(folder)
    return dataset

