ffmpeg -i P1033059.mp4 -vcodec libx264 -b:v 5M -acodec aac -b:a 256k -f flv "rtmp://localhost/live/test"