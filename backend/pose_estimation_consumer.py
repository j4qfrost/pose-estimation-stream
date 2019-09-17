import subprocess, sys, time

import json, numpy, mxnet

# from queue import Queue

import asyncio

import cv2
import pose_estimation, stream_twitch

FFMPEG= 'ffmpeg'
FFPROBE = 'ffprobe'
# f = open('blah.txt', 'w')
# f.write('jdjdjdjd')
# f.close()


def get_stream_resolution(stream_name):
	metadata = {}
	while 'streams' not in metadata:

		info = subprocess.run([FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', '../mounts/var/lib/streaming/hls/test.m3u8'],
			capture_output=True)
		out = info.stdout
		if out:
			metadata = json.loads(out.decode('utf-8'))
		time.sleep(1)

	return metadata['streams'][0]['width'], metadata['streams'][0]['height']

def get_frame_from_stream(resolution, pipe):
	width, height = resolution
	raw_image = pipe.stdout.read(width * height *3) # read 432*240*3 bytes (= 1 frame)
	# if len(raw_image) == 0:
	# 	return mxnet.nd.zeros((height, width, 3))
	# return mxnet.nd.array(numpy.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3)))
	return numpy.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))
	
async def loop_queue_frame(resolution, stream, L):
	try:
		while True:
			frame = get_frame_from_stream(resolution, stream)
			# frame = pose_estimation.process_pose_frame(frame)
			if frame is not None:
				print(f'Queuing frame... {L.qsize()}')
				await L.put(frame)
	except Exception as e:
		raise


	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	frame = pose_estimation.process_pose_frame(frame)
	# 	if frame is not None:
	# 		print(L.qsize())
	# 		await L.put(frame)

async def save_image(L):
	print('working')
	if L.qsize() == 0:
		await asyncio.sleep(1)
		await save_image(L)
	img = await L.get_nowait()
	print(L.qsize())
	cv2.imwrite('test.jpg', img)
	L.task_done()

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
	L = asyncio.Queue(maxsize=5)
	tasks = []
	tasks.append(asyncio.create_task(stream_twitch.loop_send_frame('live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY', resolution, L)))
	# tasks.append(asyncio.create_task(save_image(L)))
	tasks.append(asyncio.create_task(loop_queue_frame(resolution, stream, L)))
	await asyncio.gather(*tasks)

	# while True:
	# 	frame = get_frame_from_stream(resolution, stream)
	# 	frame = pose_estimation.process_pose_frame(frame)

	# 	if frame is not None:
	# 		L.put(frame)
	
if __name__ == '__main__':
	asyncio.run(main(sys.argv))