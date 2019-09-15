import subprocess, sys, time

import json, numpy, mxnet

# from queue import Queue

import asyncio

import pose_estimation, stream_twitch

FFMPEG= 'ffmpeg'
FFPROBE = 'ffprobe'
# f = open('blah.txt', 'w')
# f.write('jdjdjdjd')
# f.close()


def get_stream_resolution(stream_name):
	metadata = {}
	while 'streams' not in metadata:

		info = subprocess.run([FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', 'hls/test.m3u8'],
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
		sys.stdout.flush()
		print('asdads')
		return mxnet.nd.zero((height, width, 3))
	return mxnet.nd.array(numpy.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3)))

async def loop_queue_frame(resolution, stream, L):
	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	if frame is not None:
	# 		await L.put(frame)

	while True:
		frame = get_frame_from_stream(resolution, stream)
		frame = pose_estimation.process_pose_frame(frame)

		if frame is not None:
			L.put(frame)
	

async def main(argv):
	stream_name = argv[1]
	resolution = get_stream_resolution(stream_name)
	stream = subprocess.Popen([FFMPEG, '-i', stream_name,
		'-loglevel', 'quiet', # no text output
		'-an',   # disable audio
		'-f', 'image2pipe',
		'-pix_fmt', 'bgr24',
		'-vcodec', 'rawvideo', '-'],
		stdout = subprocess.PIPE)
	L = asyncio.Queue()
	stream_task = asyncio.create_task(loop_queue_frame(resolution, stream, L))
	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	frame = pose_estimation.process_pose_frame(frame)

	# 	if frame is not None:
	# 		L.put(frame)
	
	await stream_twitch.loop_send_frame('live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY', resolution, L)

if __name__ == '__main__':
	asyncio.run(main(sys.argv))