import subprocess, sys, time, os

import json, numpy

import cv2
from twitchstream.outputvideo import TwitchBufferedOutputStream
from pose_estimation import PoseProcessor

import tensorflow as tf
import posenet

FFMPEG= 'ffmpeg'
FFPROBE = 'ffprobe'

def get_stream_resolution(stream_name):
	metadata = {}
	while 'streams' not in metadata:
		info = subprocess.run([FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', stream_name],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = info.stdout
		if out:
			metadata = json.loads(out.decode('utf-8'))
		time.sleep(1)

	print('Grabbed resolution!')
	return metadata['streams'][0]['width'], metadata['streams'][0]['height']

def get_frame_from_stream(resolution, pipe):
	width, height = resolution
	raw_image = pipe.stdout.read(width * height * 3) # read 432*240*3 bytes (= 1 frame)
	if len(raw_image) == 0:
		return None
	return numpy.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))
	
def loop_send_frame(streamkey, resolution, stream, pose_processor):
	width, height = resolution
	try:
		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = 4
		config.inter_op_parallelism_threads = 4
		with TwitchBufferedOutputStream(
            twitch_stream_key=streamkey,
            width=width,
            height=height,
            fps=15.,
            enable_audio=False,
            verbose=False) as videostream:
			with tf.Session(config=config) as sess:
				model_cfg, model_outputs = posenet.load_model(3, sess)

				frame = tf.placeholder(tf.uint8, shape=(height, width, 3))
				input_image = tf.placeholder(tf.uint8, shape=(1, height + 1, width + 1, 3))

				while True:
					frame = get_frame_from_stream(resolution, stream)
					if frame is not None:
						start = time.time()
						output_stride = model_cfg['output_stride']

						input_image, frame, output_scale = posenet.process_input(
								frame, output_stride=output_stride)
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
							max_pose_detections=1, min_pose_score=0.10)

						keypoint_coords *= output_scale

						frame = posenet.draw_skel_and_kp(
								frame, pose_scores, keypoint_scores, keypoint_coords,
								min_pose_score=0.10, min_part_score=0.10)
						videostream.send_video_frame(frame)
						print(time.time() - start)
	except Exception as e:
		raise

def save_image(img):
	cv2.imwrite('test.jpg', img)

def main(stream_name):
	print('Starting program...')
	# stream_name = argv[1]
	pose_processor = PoseProcessor('tf')
	resolution = get_stream_resolution(stream_name)
	stream = subprocess.Popen([FFMPEG, '-i', stream_name,
		'-loglevel', 'quiet', # no text output
		'-an',   # disable audio
		'-f', 'image2pipe',
		'-pix_fmt', 'bgr24',
		'-vcodec', 'rawvideo', '-'],
		stdout = subprocess.PIPE, stderr=subprocess.PIPE)
	loop_send_frame('live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY', resolution, stream, pose_processor)

	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	frame = pose_estimation.process_pose_frame(frame)

	# 	if frame is not None:
	# 		L.put(frame)
	
if __name__ == '__main__':
	main(sys.argv[1])