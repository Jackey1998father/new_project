import os
import json
import numpy as np
import cv2
import radiate





def gen_boundingbox(bbox, angle):
    theta = np.deg2rad(-angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    points = np.array([[bbox[0], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                       [bbox[0], bbox[1] + bbox[3]]]).T

    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    T = np.array([[cx], [cy]])

    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)

    min_x = np.min(points[0, :])
    min_y = np.min(points[1, :])
    max_x = np.max(points[0, :])
    max_y = np.max(points[1, :])

    return min_x, min_y, max_x, max_y

def get_radar_dicts(root_dir,folders,PROPOSAL_GENERATOR = "RRPN"):
    dataset_dicts = []
    idd = 0
    folder_size = len(folders)

    # 每个天气场景下的数据集
    for folder in folders:
        radar_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
        annotation_path = os.path.join(root_dir,
                                       folder, 'annotations', 'annotations.json')
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)

        radar_files = os.listdir(radar_folder)
        radar_files.sort()

        # 针对每个场景下的每一张图片创建标签数据结构
        for frame_number in range(len(radar_files)):
            record = {}
            objs = []
            bb_created = False
            idd += 1
            filename = os.path.join(
                radar_folder, radar_files[frame_number])

            if (not os.path.isfile(filename)):
                print(filename)
                continue
            record["file_name"] = filename
            record["image_id"] = idd
            record["height"] = 1152
            record["width"] = 1152

            for object in annotation:
                if (object['bboxes'][frame_number]):
                    class_obj = object['class_name']
                    if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                        bbox = object['bboxes'][frame_number]['position']
                        angle = object['bboxes'][frame_number]['rotation']
                        bb_created = True
                        if PROPOSAL_GENERATOR == "RRPN":
                            cx = bbox[0] + bbox[2] / 2
                            cy = bbox[1] + bbox[3] / 2
                            wid = bbox[2]
                            hei = bbox[3]
                            obj = {
                                "id":object["id"],
                                "bbox": {"position": [cx, cy, wid, hei], "rotation": angle},
                                "class_name": class_obj,
                                "iscrowd": 0
                            }
                        else:
                            xmin, ymin, xmax, ymax = gen_boundingbox(
                                bbox, angle)
                            obj = {
                                "id": object["id"],
                                "bbox": {"position":[xmin, ymin, xmax, ymax], "rotation":0},
                                "class_name": class_obj,
                                "iscrowd": 0
                            }

                        objs.append(obj)
            if bb_created:
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

if __name__ == '__main__':
    # E:\new_project\radiate_sdk\data\radiate
    dataset_dicts = get_radar_dicts("E:\\new_project\\radiate_sdk\data\\radiate", ["tiny_foggy"])
    print(dataset_dicts)
    seq = radiate.Sequence(dataset_dicts[0]["file_name"].split("Navtech_Cartesian")[0])
    for obj in range(len(dataset_dicts)):
        radar_img = cv2.imread(dataset_dicts[obj]["file_name"])
        radar = seq.vis(radar_img, dataset_dicts[obj]["annotations"])
        cv2.imshow("radar", radar)
        cv2.waitKey(1000)


