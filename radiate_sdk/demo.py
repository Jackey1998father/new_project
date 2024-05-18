import radiate
import cv2
import os

# path to the sequence
root_path = 'data/radiate/'
sequence_name = 'tiny_foggy'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name))

# play sequence
# for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
print(seq.init_timestamp)
print("------------------------------------------------")
output = seq.get_from_timestamp(seq.init_timestamp)


lidar_vis = seq.vis(output['sensors']['lidar_bev_image'], output['annotations']['lidar_bev_image'])
cv2.imshow('lidar_image', lidar_vis)


init_angle = output['annotations']['radar_cartesian']
# 角度归零调整
for angle in init_angle:
    angle["bbox"]["rotation"] = 0
# radar上框出角度为零的目标
radar = seq.vis(output['sensors']['radar_cartesian'], init_angle)
cv2.imshow('radar_image', radar)


# 将radar目标狂转化为相机角度目标框
box_3d = seq.project_bboxes_to_camera(output['annotations']['radar_cartesian'],seq.calib.right_cam_mat,
                                                seq.calib.RadarToRight)
camera = seq.vis_bbox_cam(output['sensors']['camera_right_rect'],box_3d)
cv2.imshow('camera_left_rect', camera)



cv2.waitKey(5000)

    