# Rainwave Streamer

A wretched hive of scum and AI sloppery.

A mostly-AI-fueled Python application that uses gstreamer to decode, apply gain, selectively crossfade, and send music from Rainwave's backend to an Icecast server.

Tuned strictly for Rainwave's purpose.

Debian requirements:

```sh
apt install ffmpeg libgirepository-2.0-dev libcairo2-dev pkg-config python3-dev gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav
```
