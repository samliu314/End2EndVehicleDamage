/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <algorithm>
 #include <cassert>
 #include <cmath>
 #include <cstring>
 #include <fstream>
 #include <iostream>
 #include <unordered_map>
 #include "nvdsinfer_custom_impl.h"
 
 float clamp(const float val, const float minVal, const float maxVal)
 {
     assert(minVal <= maxVal);
     return std::min(maxVal, std::max(minVal, val));
 }
 
 extern "C" bool NvDsInferParseCustomYoloV4(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList);
 
 extern "C" bool NvDsInferParseCustomYoloV5(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList);
 
 extern "C" bool NvDsInferParseCustomYoloV8(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList);
 
 /* YOLOv4 implementations */
 static NvDsInferParseObjectInfo convertBBoxYoloV4(float x1, float y1, float x2,
                                     float y2, const uint& netW, const uint& netH)
 {
     NvDsInferParseObjectInfo b;
     // Restore coordinates to network input resolution
     x1 = x1 * netW;
     y1 = y1 * netH;
     x2 = x2 * netW;
     y2 = y2 * netH;
 
     x1 = clamp(x1, 0, netW);
     y1 = clamp(y1, 0, netH);
     x2 = clamp(x2, 0, netW);
     y2 = clamp(y2, 0, netH);
 
     b.left = x1;
     b.width = clamp(x2 - x1, 0, netW);
     b.top = y1;
     b.height = clamp(y2 - y1, 0, netH);
 
     return b;
 }
 
 static NvDsInferParseObjectInfo convertBBoxYoloV5(float xc, float yc, float w,
                                     float h, const uint& netW, const uint& netH)
 {
     NvDsInferParseObjectInfo b;
 
     float x = clamp(xc - w / 2, 0, netW);
     float y = clamp(yc - h / 2, 0, netH);
     w = clamp(w, 0, netW);
     h = clamp(h, 0, netH);
 
     b.left = x;
     b.width = w;
     b.top = y;
     b.height = h;
 
     return b;
 }
 
 static void addBBoxProposal(
     const int maxIndex, const float maxProb,
     std::vector<NvDsInferParseObjectInfo>& binfo, NvDsInferParseObjectInfo& bbi)
 {
     if (bbi.width < 1 || bbi.height < 1) return;
 
     bbi.detectionConfidence = maxProb;
     bbi.classId = maxIndex;
     binfo.push_back(bbi);
 }
 
 static std::vector<NvDsInferParseObjectInfo> decodeYoloV4Tensor(
     const float* boxes, const float* scores,
     const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
     const uint& netW, const uint& netH)
 {
     std::vector<NvDsInferParseObjectInfo> binfo;
 
     uint bbox_location = 0;
     uint score_location = 0;
     for (uint b = 0; b < num_bboxes; ++b)
     {
         float bx1 = boxes[bbox_location];
         float by1 = boxes[bbox_location + 1];
         float bx2 = boxes[bbox_location + 2];
         float by2 = boxes[bbox_location + 3];
 
         float maxProb = 0.0f;
         int maxIndex = -1;
 
         for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
         {
             float prob = scores[score_location + c];
             if (prob > maxProb)
             {
                 maxProb = prob;
                 maxIndex = c;
             }
         }
 
         if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
         {
             NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx1, by1, bx2, by2, netW, netH);
             addBBoxProposal(maxIndex, maxProb, binfo, bbi);
         }
 
         bbox_location += 4;
         score_location += detectionParams.numClassesConfigured;
     }
 
     return binfo;
 }
 
 
 static std::vector<NvDsInferParseObjectInfo> decodeYoloV5Tensor(
     const float* boxes,
     const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
     const uint& netW, const uint& netH)
 {
     std::vector<NvDsInferParseObjectInfo> binfo;
 
     uint location = 0;
     for (uint b = 0; b < num_bboxes; ++b)
     {
         float bxc = boxes[location];
         float byc = boxes[location + 1];
         float bw = boxes[location + 2];
         float bh = boxes[location + 3];
 
         float maxProb = 0.0f;
         int maxIndex = -1;
 
         float obj_conf = boxes[location + 4];
 
         for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
         {
             float prob = boxes[location + 5 + c] * obj_conf;
             if (prob > maxProb)
             {
                 maxProb = prob;
                 maxIndex = c;
             }
         }
 
         if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
         {
             NvDsInferParseObjectInfo bbi = convertBBoxYoloV5(bxc, byc, bw, bh, netW, netH);
             addBBoxProposal(maxIndex, maxProb, binfo, bbi);
         }
 
         location += (detectionParams.numClassesConfigured + 5);
     }
 
     return binfo;
 }
 
 static std::vector<NvDsInferParseObjectInfo> decodeYoloV8Tensor(
     const float* boxes,
     const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
     const uint& netW, const uint& netH){
 
     std::vector<NvDsInferParseObjectInfo> binfo;
 
     uint location = 0;
     for (uint b = 0; b < num_bboxes; ++b)
     {
         float bxc = boxes[location];
         float byc = boxes[location + 1 * num_bboxes];
         float bw = boxes[location + 2 * num_bboxes];
         float bh = boxes[location + 3 * num_bboxes];
 
         float maxProb = 0.0f;
         int maxIndex = -1;
 
         for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
         {
             float prob = boxes[location + (4 + c) * num_bboxes];
             if (prob > maxProb)
             {
                 maxProb = prob;
                 maxIndex = c;
             }
         }
         if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
         {
             NvDsInferParseObjectInfo bbi = convertBBoxYoloV5(bxc, byc, bw, bh, netW, netH);
             addBBoxProposal(maxIndex, maxProb, binfo, bbi);
         }
 
         location += 1;
     }
 
     return binfo;
 
 }
 
 
 extern "C" bool NvDsInferParseCustomYoloV4(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList)
 {
     auto is_debug = std::getenv("YOLO_DEBUG") == "1";
 
     std::vector<NvDsInferParseObjectInfo> objects;
 
     const NvDsInferLayerInfo &boxes = outputLayersInfo[0]; // num_boxes x 4
     const NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes
 
     // 3 dimensional: [num_boxes, 1, 4]
     assert(boxes.inferDims.numDims == 3);
     // 2 dimensional: [num_boxes, num_classes]
     assert(scores.inferDims.numDims == 2);
 
     // The second dimension should be num_classes
     assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);
 
     uint num_bboxes = boxes.inferDims.d[0];
 
     auto outObjs = decodeYoloV4Tensor(
         (const float*)(boxes.buffer), (const float*)(scores.buffer), num_bboxes, detectionParams,
         networkInfo.width, networkInfo.height);
 
     if (is_debug) {
         std::cout << "Number of Classes " << scores.inferDims.d[1] << std::endl;
         std::cout << "Number of Detected Object " << num_bboxes << "  " << outObjs.size() << std::endl;
     }
 
     objects.insert(objects.end(), outObjs.begin(), outObjs.end());
 
     objectList = objects;
 
     return true;
 }
 /* YOLOv4 implementations end*/
 
 extern "C" bool NvDsInferParseCustomYoloV5(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList)
 {
     auto is_debug = std::getenv("YOLO_DEBUG") == "1";
 
     std::vector<NvDsInferParseObjectInfo> objects;
 
     // outputLayersInfo may output other than the bbox prediction
     // Here we only use the last dimension as boxes output
     const NvDsInferLayerInfo &boxes = outputLayersInfo.back();
     const unsigned int boxesNumDim = boxes.inferDims.numDims;
 
     if (is_debug) {
         std::cout << "Output Shape: " << std::endl;
         for (auto &temp: outputLayersInfo) {
             for (auto i = 0; i < temp.inferDims.numDims; i++) {
                 std::cout << temp.inferDims.d[i] << ", ";
             }
             std::cout << std::endl;
         }
     }
 
     // Our box will always be 2 dimensional: [num_boxes, num_classes + 5]
     assert(boxesNumDim == 2);
 
     uint num_bboxes = boxes.inferDims.d[boxesNumDim - 2];
 
     assert(boxes.inferDims.d[boxesNumDim - 1] - 5 == detectionParams.numClassesConfigured);
 
     // Raw buffer of network output will be decoded to bounding box coordinate here
     // With respect to the network width & height
     auto outObjs = decodeYoloV5Tensor((const float*)(boxes.buffer), num_bboxes, detectionParams,
             networkInfo.width, networkInfo.height);
 
     objects.insert(objects.end(), outObjs.begin(), outObjs.end());
 
     objectList = objects;
 
     return true;
 }
 
 extern "C" bool NvDsInferParseCustomYoloV8(
     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
     NvDsInferNetworkInfo const& networkInfo,
     NvDsInferParseDetectionParams const& detectionParams,
     std::vector<NvDsInferParseObjectInfo>& objectList)
 {
     auto is_debug = std::getenv("YOLO_DEBUG") == "1";
 
     std::vector<NvDsInferParseObjectInfo> objects;
 
     const NvDsInferLayerInfo &boxes = outputLayersInfo[0];
 
     // 2 dimensional: [num_channels, num_boxes]
     assert(boxes.inferDims.numDims == 2);
 
     uint num_bboxes = boxes.inferDims.d[1];
 
     auto outObjs = decodeYoloV8Tensor(
         (const float*)(boxes.buffer), num_bboxes, detectionParams,
         networkInfo.width, networkInfo.height);
 
     if (is_debug) {
         std::cout << "Number of Detected Object " << num_bboxes << "  " << outObjs.size() << std::endl;
     }
 
     objects.insert(objects.end(), outObjs.begin(), outObjs.end());
 
     objectList = objects;
 
     return true;
 }
 
 /* Check that the custom function has been defined correctly */
 CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
 CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
 CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
 