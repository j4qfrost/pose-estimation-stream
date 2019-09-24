import time

class PoseProcessor:
	"""docstring for PoseProcessor"""
	def __init__(self, framework):
		super(PoseProcessor, self).__init__()
		self.framework = framework
		if framework == 'tf':
			self.process_pose_frame = build_pose_frame_function_tf()
		if framework == 'gluon':
			self.process_pose_frame = build_pose_frame_function_gluon()

def build_pose_frame_function_tf():
	import tensorflow as tf
	import posenet
	def process_pose_frame(np_frame, resolution):
		frame = None
		with tf.Session() as sess:
			model_cfg, model_outputs = posenet.load_model(0, sess)
			output_stride = model_cfg['output_stride']

			input_image, draw_image, output_scale = posenet.process_input(
					np_frame, output_stride=output_stride)

			heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
				model_outputs,
				feed_dict={'image:0': input_image}
			)

			pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
				heatmaps_result.squeeze(axis=0),
				offsets_result.squeeze(axis=0),
				displacement_fwd_result.squeeze(axis=0),
				displacement_bwd_result.squeeze(axis=0),
				output_stride=output_stride,
				max_pose_detections=1,
				min_pose_score=0.25)

			keypoint_coords *= output_scale

			frame = posenet.draw_skel_and_kp(
					draw_image, pose_scores, keypoint_scores, keypoint_coords,
					min_pose_score=0.25, min_part_score=0.25)
				

		return frame


	return process_pose_frame

def build_pose_frame_function_gluon():
	from gluoncv import model_zoo, data, utils
	from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
	from gluoncv.utils.viz import cv_plot_keypoints
	import mxnet
	import cv2

	detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
	pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

	gpu = mxnet.gpu()

	detector.reset_class(["person"], reuse_weights=['person'])
	def process_pose_frame(np_frame, resolution):
		width, height = resolution
		if np_frame is None:
		 	return mxnet.nd.zeros((height, width, 3), ctx=gpu)
		
		frame = mxnet.nd.array(np_frame, ctx=gpu)
		x, img = data.transforms.presets.yolo.transform_test(frame, short=512)

		class_IDs, scores, bounding_boxs = detector(x)

		pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

		if pose_input is None:
			return

		predicted_heatmap = pose_net(pose_input)
		pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
		img = cv_plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2, scale=1.0, **kwargs)
		print(img.size)
		# for j in range(len(pred_coords)):
		# 	for i in range(len(pred_coords[0])):
		# 		x, y = pred_coords[j][i].astype(int).asnumpy()
		# 		cv2.circle(img, (x,y), 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
		return img
	return process_pose_frame

def process_pose_frame_torch(np_frame, resolution):
	pass