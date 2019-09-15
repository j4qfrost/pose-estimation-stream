from twitchstream.outputvideo import TwitchBufferedOutputStream
import asyncio, sys

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

        # The main loop to create videos
        while True:
            # If there are not enough video frames left,
            # add some more.
            if videostream.get_video_frame_buffer_state() < 30:
                frame = await L.get()
                if frame.size > 0:
                    videostream.send_video_frame(frame)
            else:
                sys.stdout.flush()
                print(L.qsize())
                await asyncio.sleep(0.001)
            sys.stdout.flush()
            print(L.qsize())