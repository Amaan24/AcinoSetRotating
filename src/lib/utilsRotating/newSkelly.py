import os
from typing import Dict
import pickle
import matplotlib.pyplot as plt


def load_skeleton(skel_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict


def save_skeleton(path, skelDict):
    with open(path, 'wb') as f:
        pickle.dump(skelDict, f)


def plot_skelly() -> None:
    f = plt.figure()
    a = plt.axes(projection='3d')

    pose_dict = {}

    for i in markers:
        point = [positions[i][0], positions[i][1], positions[i][2]]
        print(point)
        pose_dict[markers[markers.index(i)]] = point
        a.scatter(point[0], point[1], point[2], color="red")
        a.text(point[0], point[1], point[2], i, size=8, zorder=1, color='k')


    for link in links:
        if len(link) > 1:
            a.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                     [pose_dict[link[0]][1], pose_dict[link[1]][1]],
                     [pose_dict[link[0]][2], pose_dict[link[1]][2]], color="black", )

    a.set_xlabel('x')
    a.set_ylabel('y')
    a.set_zlabel('z')

    a.set_xlim(-2, 2)
    a.set_ylim(-2, 2)
    a.set_zlim(0, 2.5)
    plt.show()


links = [['forehead', 'chin'],
         ['forehead', 'neck'],
         ['neck', 'shoulder1'],
         ['neck', 'shoulder2'],
         ['shoulder1', 'hip1'],
         ['shoulder2', 'hip2'],
         ['shoulder1', 'elbow1'],
         ['shoulder2', 'elbow2'],
         ['elbow1', 'wrist1'],
         ['elbow2', 'wrist2'],
         ['hip1', 'hip2'],
         ['hip1', 'knee1'],
         ['hip2', 'knee2'],
         ['knee1', 'ankle1'],
         ['knee2', 'ankle2']]

dofs = {'forehead': [1, 1, 1],
        'chin': [0, 0, 0],
        'neck': [1, 1, 1],
        'shoulder1': [1, 1, 1],
        'shoulder2': [1, 1, 1],
        'elbow1': [0, 1, 0],
        'elbow2': [0, 1, 0],
        'wrist1': [0, 0, 0],
        'wrist2': [0, 0, 0],
        'hip1': [1, 1, 1],
        'hip2': [1, 1, 1],
        'knee1': [0, 1, 0],
        'knee2': [0, 1, 0],
        'ankle1': [0, 0, 0],
        'ankle2': [0, 0, 0]}


positions= {
            'forehead': [0.26898,  6.636344,  0.816441],
            'neck': [0.261187,  6.572296,  0.573736],
            'chin': [0.263661,  6.627421,  0.654366],
            'shoulder1': [0.457773,  6.759311,  0.576362],
            'shoulder2': [0.078325,  6.570917,  0.585779],
            'elbow1': [0.616712,  7.046775,  0.341119],
            'elbow2': [-0.066443,  6.639662,  0.359051],
            'wrist1': [0.472888,  6.699009,  0.12238],
            'wrist2': [0.001705,  6.733185,  0.149866],
            'hip1': [0.356712,  6.70816, -0.06091],
            'hip2': [0.098365,  6.704401, -0.051397],
            'knee1': [0.373637,  6.857199, -0.484482],
            'knee2': [0.06255,  6.695557, -0.460909],
            'ankle1': [0.365857,  6.890354, -0.822666], 
            'ankle2': [0.062193,  6.857276, -0.792951],
 }

'''
positions = {'forehead': [0.0, 0.00, 1.71],
             'chin': [0.0, 0.00, 1.53],
             'neck': [0.0, 0.0, 1.44],
             'shoulder1': [-0.2, 0.0, 1.44],
             'shoulder2': [0.2, 0.0, 1.44],
             'elbow1': [-0.45, 0.0, 1.44],
             'elbow2': [0.45, 0.0, 1.44],
             'wrist1': [-0.7, 0.0, 1.44],
             'wrist2': [0.7, 0.0, 1.44],
             'hip1': [-0.1, 0.0, 0.92],
             'hip2': [0.1, 0.0, 0.92],
             'knee1': [-0.1, 0.0, 0.48],
             'knee2': [0.1, 0.0, 0.48],
             'ankle1': [-0.1, 0.0, 0.0],
             'ankle2': [0.1, 0.0, 0.0]}

positions = {'chin': [2, 1.79, 1.53],
             'forehead': [2, 1.81, 1.71],
             'neck': [2, 2, 1.44],
             'shoulder1': [2-0.225, 2, 1.44],
             'shoulder2': [2+0.225, 2, 1.44],
             'elbow1': [2-0.545, 2, 1.44],
             'elbow2': [2+0.545, 2, 1.44],
             'wrist1': [2-0.845, 2, 1.44],
             'wrist2': [2+0.845, 2, 1.44],
             'hip1': [2-0.125, 2, 0.92],
             'hip2': [2+0.125, 2, 0.92],
             'knee1': [2-0.125, 2, 0.48],
             'knee2': [2+0.125, 2, 0.48],
             'ankle1': [2-0.125, 2, 0.0],
             'ankle2': [2+0.125, 2, 0.0]}
'''

markers = ['forehead',
           'chin',
           'neck',
           'shoulder1',
           'shoulder2',
           'hip1',
           'hip2',
           'elbow1',
           'elbow2',
           'wrist1',
           'wrist2',
           'knee1',
           'knee2',
           'ankle1',
           'ankle2']

skel_dict = {'links': links, 'dofs': dofs, 'positions': positions, 'markers': markers}
plot_skelly()

skelly = input("Enter skeleton name (name.pickle):")
skelly_dir = os.path.join("C://Users//user-pc//Desktop/AcinoSetRotating//skeletons", skelly,)
save_skeleton(skelly_dir, skel_dict)
