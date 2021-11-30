"""
plot with matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import cv2
import json


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    ax = plt.gca()
    for i, seg in enumerate(segments):
        seg = np.expand_dims(seg, axis=0)
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth / (i+1), alpha=alpha / (i+1))

        ax.add_collection(lc)

    # return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def main():
    N = 10
    np.random.seed(101)
    x = np.random.rand(N)
    y = np.random.rand(N)
    fig, ax = plt.subplots()

    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=3)

    plt.show()


def main2():
    img_path = 'data/COCO_test2015_000000387726.jpg'
    im = cv2.imread(img_path)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch, xywh (xy is top left)
    rect = Rectangle((50, 100), 50, 100, linewidth=5, edgecolor='r',
                     facecolor='none', alpha=1)

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.plot(75, 150, 'ro')
    plt.show()


def find_question():
    target_question = 'Are there any trees in this picture?'
    tasks = ['train2014', 'test2015', 'test-dev2015', 'val2014']

    for t in tasks:
        q1 = json.load(
            open(f'data/v2_OpenEnded_mscoco_{t}_questions.json'))
        all_q = [q['question'] for q in q1['questions']]
        qs_arr = np.array(all_q)
        indices = np.where(qs_arr == target_question)
        print(indices)
        for idx in indices[0]:
            q_dict = q1['questions'][idx]
            print(f"image_id: {q_dict['image_id']}")
    # with open(f'data/{task1}_q.txt', 'w') as f:
    #     f.writelines(qs)


def read_adj():
    adj_file = 'data/adj_mat.npz'
    npzfile = np.load(adj_file)
    a = npzfile['arr_0']
    a0_sorted = np.sort(a[0], axis=0)
    a0 = a0_sorted
    print(a0)


if __name__ == '__main__':
    # main()
    # main2()
    # find_question()
    read_adj()
