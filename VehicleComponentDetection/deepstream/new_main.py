#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import os
import sys
sys.path.append("../")
from common.bus_call import bus_call
from common.platform_info import PlatformInfo
import pyds
import platform
import math
import time
from ctypes import *
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
import configparser
import datetime

import argparse

# MAX_DISPLAY_LEN = 64
# PGIE_CLASS_ID_VEHICLE = 0
# PGIE_CLASS_ID_BICYCLE = 1
# PGIE_CLASS_ID_PERSON = 2
# PGIE_CLASS_ID_ROADSIGN = 3
# MUXER_OUTPUT_WIDTH = 1920
# MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
# GST_CAPS_FEATURES_NVMM = "memory:NVMM"
# OSD_PROCESS_MODE = 0
# OSD_DISPLAY_TEXT = 0
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "/home/saml/DamageDetection/deepstream"
# pgie_src_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.


def pgie_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        print("Unable to get batch metadata")
        return Gst.PadProbeReturn.OK
        
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        if frame_number % 100 == 0:
            print(f"Frame Number={frame_number}, Source ID={frame_meta.source_id}, Batch ID={frame_meta.batch_id}")
        
        # # Get object metadata
        # l_obj = frame_meta.obj_meta_list
        # num_rects = 0
        # while l_obj is not None:
        #     try:
        #         # Cast to object metadata
        #         obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        #     except StopIteration:
        #         break
                
        #     # Update counter
        #     num_rects += 1
            
        #     # Get class name from labels list
        #     class_name = "Unknown"
        #     if obj_meta.class_id < 15:  # Number of classes in labels.txt
        #         with open("labels.txt", "r") as f:
        #             labels = f.read().strip().split("\n")
        #             if obj_meta.class_id < len(labels):
        #                 class_name = labels[obj_meta.class_id]
            
        #     # Display confidence and class id
        #     print(f"Object {num_rects}: class_id={obj_meta.class_id} ({class_name}), confidence={obj_meta.confidence:.2f}")
        #     print(f"  Rect: X={obj_meta.rect_params.left}, Y={obj_meta.rect_params.top}, W={obj_meta.rect_params.width}, H={obj_meta.rect_params.height}")
            
        #     # Add display text for the object
        #     txt_params = obj_meta.text_params
        #     txt_params.display_text = f"{class_name}: {obj_meta.confidence:.2f}"
        #     txt_params.font_params.font_name = "Arial"
        #     txt_params.font_params.font_size = 12
        #     txt_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow
        #     txt_params.set_bg_clr = 1
        #     txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)  # Semi-transparent black
            
        #     # Set rectangle color based on class ID (for visual distinction)
        #     obj_meta.rect_params.border_width = 2
        #     color_param = obj_meta.rect_params.border_color
        #     # Cycle through some distinct colors based on class ID
        #     colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1)]
        #     color_idx = obj_meta.class_id % len(colors)
        #     r, g, b = colors[color_idx]
        #     color_param.set(r, g, b, 1.0)
            
        #     try:
        #         l_obj = l_obj.next
        #     except StopIteration:
        #         break
                        
        if ts_from_rtsp:
            ts = frame_meta.ntp_timestamp/1000000000 # Retrieve timestamp, put decimal in proper position for Unix format
            print("RTSP Timestamp:",datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) # Convert timestamp to UTC

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if ts_from_rtsp:
        if name.find("source") != -1:
            pyds.configure_source_for_ntp_sync(hash(Object))


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
        return None

    # Check if CSI camera is requested
    if uri.startswith("csi://"):
        print("Creating CSI camera source")
        # Parse CSI camera parameters from URI
        # Format: csi://sensor_id
        # Example: csi://0
        try:
            parts = uri.split("/")
            sensor_id = int(parts[2]) if len(parts) > 2 else 0
        except (IndexError, ValueError):
            sys.stderr.write(" Error parsing CSI camera parameters \n")
            return None

        # Create nvarguscamerasrc for CSI camera
        print(f"CSI camera: sensor_id={sensor_id}")
        src_elem = Gst.ElementFactory.make("nvarguscamerasrc", "src-elem")
        if not src_elem:
            sys.stderr.write(" Unable to create nvarguscamerasrc \n")
            return None

        # Configure camera properties
        src_elem.set_property("sensor-id", sensor_id)
        
        # Create nvvidconv to convert from NV12 to RGBA
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")
            return None
            
        # Create capsfilter
        caps = Gst.ElementFactory.make("capsfilter", "filter")
        if not caps:
            sys.stderr.write(" Unable to create capsfilter \n")
            return None
            
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        # Add elements to bin
        Gst.Bin.add(nbin, src_elem)
        Gst.Bin.add(nbin, caps)
        Gst.Bin.add(nbin, nvvidconv)

        # Link elements
        src_elem.link(caps)
        caps.link(nvvidconv)

        # Create ghost pad
        srcpad = nvvidconv.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Failed to get src pad from nvvidconv \n")
            return None
            
        bin_pad = nbin.add_pad(Gst.GhostPad.new("src", srcpad))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
    else:
        # Original URI source handling
        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
            return None
        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(
            Gst.GhostPad.new_no_target(
                "src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
    
    return nbin


def main(args):
    # Check input arguments
    number_sources = len(args)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Pgie \n ")
    if gie=="nvinfer":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if platform_info.is_integrated_gpu():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        #encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    sink.set_property("host", "0.0.0.0")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)

    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    
    if ts_from_rtsp:
        streammux.set_property("attach-sys-ts", 0)

    if gie=="nvinfer":
        pgie.set_property("config-file-path", "./config/pgie_config.txt")
    else:
        pgie.set_property("config-file-path", "./config/pgie_inferserver_config.txt")


    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    # Create queue elements to buffer data between pipeline elements
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    
    # Configure queue properties for better buffering
    for q in [queue1, queue2, queue3, queue4, queue5, queue6]:
        q.set_property("max-size-buffers", 4)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        q.set_property("leaky", 0)  # No leaky queue
    
    # Add all elements to the pipeline
    pipeline.add(pgie)
    pipeline.add(queue1)
    pipeline.add(tiler)
    pipeline.add(queue2)
    pipeline.add(nvvidconv)
    pipeline.add(queue3)
    pipeline.add(nvosd)
    pipeline.add(queue4)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(queue5)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(queue6)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Link elements with queues in between for buffering
    streammux.link(pgie)
    pgie.link(queue1)
    queue1.link(tiler)
    tiler.link(queue2)
    queue2.link(nvvidconv)
    nvvidconv.link(queue3)
    queue3.link(nvosd)
    nvosd.link(queue4)
    queue4.link(nvvidconv_postosd)
    nvvidconv_postosd.link(queue5)
    queue5.link(caps)
    caps.link(encoder)
    encoder.link(queue6)
    queue6.link(rtppay)
    rtppay.link(sink)

    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad=pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
        % rtsp_port_num
    )

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-g", "--gie", default="nvinfer",
                  help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer','nvinferserver'])
    parser.add_argument("-c", "--codec", default="H264",
                  help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=False, dest='rtsp_ts', help="Attach NTP timestamp from RTSP source",
    )
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global gie
    global ts_from_rtsp
    gie = args.gie
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    return stream_path

if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
