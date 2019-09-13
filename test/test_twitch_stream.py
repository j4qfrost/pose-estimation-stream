from twitchstream.outputvideo import TwitchBufferedOutputStream
import argparse
import time
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-s', '--streamkey',
                          help='twitch streamkey',
                          required=True)
    args = parser.parse_args()

    # load two streams:
    # * one stream to send the video
    # * one stream to interact with the chat
    with TwitchBufferedOutputStream(
            twitch_stream_key=args.streamkey,
            width=640,
            height=480,
            fps=30.,
            enable_audio=True,
            verbose=False) as videostream:


        frame = np.zeros((480, 640, 3))
        frequency = 100
        last_phase = 0

        # The main loop to create videos
        while True:

            # Every loop, call to receive messages.
            # This is important, when it is not called,
            # Twitch will automatically log you out.
            # This call is non-blocking.


            # If there are not enough video frames left,
            # add some more.
            if videostream.get_video_frame_buffer_state() < 30:
                videostream.send_video_frame(frame)

            # If there are not enough audio fragments left,
            # add some more, but take care to stay in sync with
            # the video! Audio and video buffer separately,
            # so they will go out of sync if the number of video
            # frames does not match the number of audio samples!
            elif videostream.get_audio_buffer_state() < 30:
                x = np.linspace(last_phase,
                                last_phase +
                                frequency*2*np.pi/videostream.fps,
                                int(44100 / videostream.fps) + 1)
                last_phase = x[-1]
                audio = np.sin(x[:-1])
                videostream.send_audio(audio, audio)

            # If nothing is happening, it is okay to sleep for a while
            # and take some pressure of the CPU. But not too long, if
            # the buffers run dry, audio and video will go out of sync.
            else:
                time.sleep(.001)