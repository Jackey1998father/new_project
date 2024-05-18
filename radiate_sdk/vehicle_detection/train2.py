import os
import json



def get_radar_dicts(folders):
    idd = 0
    category_list = {"pedestrain":0,"car":1,"truck":2}
    dataset_dicts = []
    for folder in folders:
        folder_path = os.path.join(root_, folder,'Navtech_Cartesian')
        annotation_path = os.path.join(root_, folder, 'annotations')
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)



        for filename in range(len(os.listdir(folder_path))):
            idd += 1
            record = {}
            car_img = os.path.join(folder_path,folder_path(filename))
            # 将每个图片打包
            record["file_name"] = car_img
            record["image_id"] = idd
            record["height"] = 1152
            record["width"] = 1152
            anno_ = []
            for object in annotation:
                rec = {}
                if (object['bboxes'][filename]):
                    class_obj = object['class_name']
                    if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):

                        rec["bbox"] = object['bboxes'][filename]['position'].append(object['rotation'])
                        rec["bbox_mode"] = "form"
                        rec['category_id'] = category_list[class_obj]
                        rec['iscrowd'] = 0
                anno_.append(rec)
            record["annotations"] =anno_
            dataset_dicts.append(record)


if __name__ == '__main__':
    root_ = 'C:\\Users\\USER\\Desktop\\new_project\\radiate_sdk\\data\\radiate'
    get_radar_dicts = os.listdir(root_)
    get_radar_dicts(get_radar_dicts)