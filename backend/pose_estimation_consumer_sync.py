import subprocess, sys, time

import json, numpy

import cv2
from twitchstream.outputvideo import TwitchBufferedOutputStream
from pose_estimation import PoseProcessor

FFMPEG= 'ffmpeg'
FFPROBE = 'ffprobe'

def get_stream_resolution(stream_name):
	metadata = {}
	while 'streams' not in metadata:
		info = subprocess.run([FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', '../hls/test.m3u8'],
			capture_output=True)
		out = info.stdout
		if out:
			metadata = json.loads(out.decode('utf-8'))
		time.sleep(1)

	return metadata['streams'][0]['width'], metadata['streams'][0]['height']

def get_frame_from_stream(resolution, pipe):
	width, height = resolution
	raw_image = pipe.stdout.read(width * height *3) # read 432*240*3 bytes (= 1 frame)
	if len(raw_image) == 0:
		return None
	return numpy.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))
	
def loop_send_frame(streamkey, resolution, stream, pose_processor):
	videostream = TwitchBufferedOutputStream(
            twitch_stream_key=streamkey,
            width=resolution[0],
            height=resolution[1],
            fps=30.,
            enable_audio=False,
            verbose=False)
	try:
		while True:
			frame = get_frame_from_stream(resolution, stream)
			if frame is not None:
				print(frame.size)
				frame = pose_processor.process_pose_frame(frame, resolution)
				print(frame.size)
				videostream.send_video_frame(frame)
	except Exception as e:
		raise

def save_image(img):
	cv2.imwrite('test.jpg', img)

def main(argv):
	stream_name = argv[1]
	pose_processor = PoseProcessor('gluon')
	resolution = get_stream_resolution(stream_name)
	stream = subprocess.Popen([FFMPEG, '-i', stream_name,
		'-loglevel', 'quiet', # no text output
		'-an',   # disable audio
		'-f', 'image2pipe',
		'-pix_fmt', 'bgr24',
		'-vcodec', 'rawvideo', '-'],
		stdout = subprocess.PIPE)
	loop_send_frame('live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY', resolution, stream, pose_processor)

	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	frame = pose_estimation.process_pose_frame(frame)

	# 	if frame is not None:
	# 		L.put(frame)
	
if __name__ == '__main__':
	main(sys.argv)