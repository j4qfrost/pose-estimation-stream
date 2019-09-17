from twitchstream.outputvideo import TwitchBufferedOutputStream
import asyncio, sys, cv2, time

async def loop_send_frame(streamkey, resolution, L):
    # load two streams:
     # * one stream to send the video
    with TwitchBufferedOutputStream(
            twitch_stream_key=streamkey,
            width=resolution[0],
            height=resolution[1],
            fps=30.,
            enable_audio=False,
            verbose=False) as videostream:
        try:
                    # The main loop to create videos
            while True:
                # If there are not enough video frames left,
                # add some more.
                print(f'Sending frame... {L.qsize()}')
                frame = await L.get()
                if frame.size > 0:
                    videostream.send_video_frame(frame)
                L.task_done()


        except Exception as e:
            raise e
