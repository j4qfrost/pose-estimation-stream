import time

import cv2
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])

def process_pose_frame(np_frame, resolution):
	width, height = resolution
	if np_frame is None:
	 	return mxnet.nd.zeros((height, width, 3))
	
	frame = mxnet.nd.array(np_frame)
	x, img = data.transforms.presets.ssd.transform_test(frame, short=512)

	class_IDs, scores, bounding_boxs = detector(x)

	pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

	if pose_input is None:
		return

	predicted_heatmap = pose_net(pose_input)
	pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

	for j in range(len(pred_coords)):
		for i in range(len(pred_coords[0])):
			x, y = pred_coords[j][i].astype(int).asnumpy()
			cv2.circle(img, (x,y), 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
			# cv2.putText(img, tag, (x, y-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

	# for i in range(len(scores[0])):
	#     #print(class_IDs.reshape(-1))
	#     #print(scores.reshape(-1))
	#     cid = int(class_IDs[0][i].asnumpy())
	#     cname = detector.classes[cid]
	#     score = float(scores[0][i].asnumpy())
	#     if score < 0.5:
	#         break
	#     x,y,w,h = bbox =  bounding_boxs[0][i].astype(int).asnumpy()
	#     print(cid, score, bbox)
	#     tag = "{}; {:.4f}".format(cname, score)
	#     cv2.rectangle(img, (x,y), (w, h), (0, 255, 0), 2)
	#     cv2.putText(img, tag, (x, y-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

	return img