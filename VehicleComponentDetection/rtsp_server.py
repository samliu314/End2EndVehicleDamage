import gi
gi.require_version('Gst','1.0')
gi.require_version('GstRtspServer','1.0')
from gi.repository import GLib, Gst, GstRtspServer

def create_rtsp_server():
    Gst.init(None)
    mainloop = GLib.MainLoop()
    server = GstRtspServer.RTSPServer()
    mounts = server.get_mount_points()

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch('( udpsrc port=1234 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! h264parse ! rtph264pay name=pay0 )')

    mounts.add_factory("/cam", factory)
    server.attach(None)

    print("stream ready at rtsp://0.0.0.0:8554/cam")
    mainloop.run()