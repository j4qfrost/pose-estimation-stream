import subprocess, sys, time

import json, numpy, mxnet

import pose_estimation

FFMPEG= 'ffmpeg'
FFPROBE = 'ffprobe'
# f = open('blah.txt', 'w')
# f.write('jdjdjdjd')
# f.close()


def get_stream_resolution(stream_name):
	metadata = {}
	while 'streams' not in metadata:
		info = subprocess.Popen([FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', 'hls/test.m3u8'],
			stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = info.communicate()
		if out:
			metadata = json.loads(out.decode('utf-8'))
		f = open('blah.txt', 'w')
		f.write(str(err))
		f.write(str(out))
		f.close()
		time.sleep(5)

	return metadata['streams'][0]['width'], metadata['streams'][0]['height']

def get_frame_from_stream(resolution, pipe):
	width, height = resolution
	raw_image = pipe.stdout.read(width * height *3) # read 432*240*3 bytes (= 1 frame)
	return mxnet.nd.array(numpy.fromstring(raw_image, dtype='uint8').reshape((width, height, 3)))

def main(argv):
	stream_name = argv[1]
	resolution = get_stream_resolution(stream_name)
	stream = subprocess.Popen([FFMPEG, '-i', stream_name,
		'-loglevel', 'quiet', # no text output
		'-an',   # disable audio
		'-f', 'image2pipe',
		'-pix_fmt', 'bgr24',
		'-vcodec', 'rawvideo', '-'],
		stdin = subprocess.PIPE, stdout = subprocess.PIPE)
	twitch_stream = subprocess.Popen([FFMPEG, '-i', '-',
		'-vcodec', 'libx264', # no text output
		'-an',   # disable audio
		'-pix_fmt', 'yuv420p',
		'-tune', 'fastdecode', 'rtmp://live-sjc.twtich.tv/apps/live_173288790_pEOfgLFUAfocVRZdAQ1D8bUubjL4OY'],
		stdin = subprocess.PIPE, stdout = subprocess.PIPE)
	while True:
		frame = get_frame_from_stream(resolution, stream)
		frame = pose_estimation.process_pose_frame(frame)
		twitch_stream.communicate(input=frame.tobytes())

if __name__ == '__main__':
	main(sys.argv)