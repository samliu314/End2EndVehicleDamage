#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
    $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes/ -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvdsparsebbox_Yolo.cpp nvinfer_custom_lpr_parser.cpp nvdsparser_classifier.cpp

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so libnvdsinfer_custom_impl_lpr.so libnvdsinfer_custom_impl_classifier.so

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

libnvdsinfer_custom_impl_Yolo.so : nvdsparsebbox_Yolo.o
	$(CC) -o $@ $^ $(LFLAGS)

libnvdsinfer_custom_impl_lpr.so : nvinfer_custom_lpr_parser.o
	$(CC) -o $@ $^ $(LFLAGS)

libnvdsinfer_custom_impl_classifier.so : nvdsparser_classifier.o
	$(CC) -o $@ $^ $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
