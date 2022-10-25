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


def plot_results() -> None:
    """
    Plots results for the given skeleton (frame 0)
    """
    f = plt.figure()
    a = plt.axes(projection='3d')

    pose_dict = {}

    skelly_dir = os.path.join("C://Users//user-pc//Desktop/AcinoSetRotating//skeletons", ("human_no_chin.pickle"))

    skel_dict = load_skeleton(skelly_dir)
    print(skel_dict)
    # output_dir = os.path.join("C://Users//user-pc//Desktop/AcinoSetRotating//skeletons", "human_no_chin.pickle")

    # with open(output_dir, 'wb') as f:
    #    pickle.dump(skel_dict, f)

    links = skel_dict["links"]
    markers = skel_dict["markers"]

    positions = {"chin": skel_dict["positions"]["chin"], "forehead": skel_dict["positions"]["forehead"],
                 "shoulder1": skel_dict["positions"]["shoulder1"], "shoulder2": skel_dict["positions"]["shoulder2"],
                 "elbow1": skel_dict["positions"]["elbow1"], "elbow2": skel_dict["positions"]["elbow2"],
                 "hip1": skel_dict["positions"]["hip1"], "hip2": skel_dict["positions"]["hip2"],
                 "wrist1": skel_dict["positions"]["wrist1"], "wrist2": skel_dict["positions"]["wrist2"],
                 "knee1": skel_dict["positions"]["knee1"], "knee2": skel_dict["positions"]["knee2"],
                 "ankle1": skel_dict["positions"]["ankle1"], "ankle2": skel_dict["positions"]["ankle2"]}

    for i in markers:
        point = [positions[i][0], positions[i][1], positions[i][2]]
        print(point)
        pose_dict[markers[markers.index(i)]] = point
        a.scatter(point[0], point[1], point[2], color="red")

    print(pose_dict)

    for link in links:
        if len(link) > 1:
            a.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                     [pose_dict[link[0]][1], pose_dict[link[1]][1]],
                     [pose_dict[link[0]][2], pose_dict[link[1]][2]], color="black")
    plt.show()


plot_results()
