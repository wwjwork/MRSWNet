import numpy as np
import torch
from torchvision.ops import nms


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):

            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for \
                              anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])
            w = torch.sigmoid(prediction[..., 2]) 
            h = torch.sigmoid(prediction[..., 3]) 

            conf        = torch.sigmoid(prediction[..., 4])
            pred_cls    = torch.sigmoid(prediction[..., 5:])
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)
            
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1]  = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2]  = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3]  = (h.data * 2) ** 2 * anchor_h

            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):

        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]

                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


class DecodeBoxNP():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBoxNP, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            batch_size      = np.shape(input)[0]
            input_height    = np.shape(input)[2]
            input_width     = np.shape(input)[3]

            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]
            prediction = np.transpose(np.reshape(input, (batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width)), (0, 1, 3, 4, 2))

            x = self.sigmoid(prediction[..., 0])  
            y = self.sigmoid(prediction[..., 1])
            w = self.sigmoid(prediction[..., 2]) 
            h = self.sigmoid(prediction[..., 3]) 

            conf        = self.sigmoid(prediction[..., 4])
            pred_cls    = self.sigmoid(prediction[..., 5:])

            grid_x = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.linspace(0, input_width - 1, input_width), 0), input_height, axis=0), 0), batch_size * len(self.anchors_mask[i]), axis=0)
            grid_x = np.reshape(grid_x, np.shape(x))
            grid_y = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.linspace(0, input_height - 1, input_height), 0), input_width, axis=0).T, 0), batch_size * len(self.anchors_mask[i]), axis=0)
            grid_y = np.reshape(grid_y, np.shape(y))

            anchor_w = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 0], 0), batch_size, axis=0), -1), input_height * input_width, axis=-1)
            anchor_h = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 1], 0), batch_size, axis=0), -1), input_height * input_width, axis=-1)
            anchor_w = np.reshape(anchor_w, np.shape(w))
            anchor_h = np.reshape(anchor_h, np.shape(h))

            pred_boxes          = np.zeros(np.shape(prediction[..., :4]))
            pred_boxes[..., 0]  = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1]  = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2]  = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3]  = (h * 2) ** 2 * anchor_h

            _scale = np.array([input_width, input_height, input_width, input_height])
            output = np.concatenate([np.reshape(pred_boxes, (batch_size, -1, 4)) / _scale,
                                np.reshape(conf, (batch_size, -1, 1)), np.reshape(pred_cls, (batch_size, -1, self.num_classes))], -1)
            outputs.append(output)
        return outputs
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
                    
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        iou = inter_area / np.maximum(b1_area + b2_area - inter_area, 1e-6)

        return iou

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        box_corner          = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)
            class_pred = np.expand_dims(np.argmax(image_pred[:, 5:5 + num_classes], 1), -1)
            conf_mask  = np.squeeze((image_pred[:, 4] * class_conf[:, 0] >= conf_thres))

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not np.shape(image_pred)[0]:
                continue

            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)
            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                conf_sort_index     = np.argsort(detections_class[:, 4] * detections_class[:, 5])[::-1]
                detections_class    = detections_class[conf_sort_index]
                max_detections = []
                while np.shape(detections_class)[0]:
                    max_detections.append(detections_class[0:1])
                    if len(detections_class) == 1:
                        break
                    ious                = self.bbox_iou(max_detections[-1], detections_class[1:])
                    detections_class    = detections_class[1:][ious < nms_thres]
                max_detections = np.concatenate(max_detections, 0)

                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i]
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
