import argparse
import time
from pathlib import Path

import json

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import yaml
from torchvision import transforms
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_mask_conf, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


global next_flag
global global_input_points
global global_input_negative_points


def OnMouseAction(event, x, y, flags, param):
    global next_flag
    global global_input_points
    if event == cv2.EVENT_LBUTTONDOWN: 
        global_input_points.append((x, y))
        print("单击了鼠标左键") 
    if event==cv2.EVENT_LBUTTONDBLCLK:
        next_flag = 1




def segment_anything(image_without_mask, key_points, predictor, h0, w0, h1, w1):
    global next_flag
    global global_input_points
    predictor.set_image(image_without_mask)
    global_input_points, next_flag = [], 0

    #candidate_points = (key_points[4:]).reshape((18, 3))
    input_point = []
    # for i in range(18):
    #     if candidate_points[i, 2] > 0.1:
    #         x = w0 + candidate_points[i, 0]*ratio
    #         y = h0 + candidate_points[i, 1]*ratio
    #         input_point.append([x, y])
    img_marked = image_without_mask.copy()
    pts=np.array([[w0,h0], [w1,h0], [w1,h1], [w0,h1]], np.int32)
    pts=pts.reshape((-1,1,2)) 
    img_marked = cv2.polylines(img_marked, [pts], True, (0,0,255),8) 

    while True:
        cv2.imshow("test", img_marked) 
        
        cv2.waitKey()

        input_point = np.array(global_input_points)
        input_label = np.array([1]*input_point.shape[0])

        print("input points:", input_point)

        if input_point.size == 0:
            return False, image_without_mask
        
        img_marked = cv2.circle(img_marked, (global_input_points[-1][0], global_input_points[-1][1]), 1, (255, 0, 0), 3)

        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        )

        img_marked[masks[0]] = img_marked[masks[0]] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5

        if next_flag == 1:
            print("Break")
            break

    
    # if input_point.size == 0:
    #     return False, image_without_mask


    image_without_mask[masks[0] == False] = np.array([0, 0, 0])
    return True, image_without_mask

def fetch_skeleton(image, model, device, height=640):
    image = letterbox(image, height, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()

    output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)

    # print("Shape of key points:", output.shape)
    # print("key_points:", output)
    if output.size == 0:
        return output

    return output[0]



def add_mask(image, model, device, hyp, height=640):

    #downscale the image
    image = letterbox(image, height, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()
    output = model(image)

    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']

    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
    pnimg = nimg.copy()
    image_without_mask = pnimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        mask_area_prop = 1 - np.sum(1-one_mask)/(pnimg.shape[0]*pnimg.shape[1])
        if conf < 0.50 or mask_area_prop < 0.20:
            continue    

        pnimg[one_mask == False] = np.array([0, 0, 0])


    return pnimg, image_without_mask


def save_sub_image(img, xyxy, frame_idx, bbx_idx, device, model_seg, model_pose, predictor, hyp, aim_shape=[640, 320]):

    orig_img = img.copy()

    save_path = "dataset/processed_images/"+str(frame_idx).zfill(5)+"_"+str(bbx_idx).zfill(2) + ".jpg"
    h_w_ratio = aim_shape[0]/aim_shape[1]
    height, width = xyxy[3].cpu() - xyxy[1].cpu(), xyxy[2].cpu() - xyxy[0].cpu()
    center_y, center_x = (xyxy[3].cpu() + xyxy[1].cpu())/2, (xyxy[2].cpu() + xyxy[0].cpu())/2

    ##Crop
    if height/width > h_w_ratio:
        width = height/h_w_ratio
    else:
        height = width*h_w_ratio

    out_of_range = ((center_x-width/2) < 0) or ((center_x+width/2) > img.shape[1]-1) or \
                    ((center_y-height/2) < 0) or ((center_y+height/2) > img.shape[0]-1)

    too_small = (height < 640)
    

    if out_of_range:
        print("Warning: Out of Range!", save_path)
        return False, {}
    
    if too_small:
        print("Warning: Too Small!", save_path)
        return False, {}

    
    img, image_without_mask = add_mask(img[int(center_y-height/2):int(center_y+height/2), int(center_x-width/2):int(center_x+width/2)], model_seg, device, hyp, height=aim_shape[0])
                   
    print("subimage saved", "sub_image_path: ", save_path)


    key_points = fetch_skeleton(img, model_pose, device)
    if key_points.size == 0:
        return False, {}

    success, image = segment_anything(orig_img, key_points, predictor, int(center_y-height/2), int(center_x-width/2), int(center_y+height/2), int(center_x+width/2))

    if not success:
        return False, {}

    image = cv2.resize(image[int(center_y-height/2):int(center_y+height/2), int(center_x-width/2):int(center_x+width/2)], (320, 640))

    cv2.imwrite(save_path, image)
    info_dict = {
        "sub_image_path": save_path,
        "height": aim_shape[0],
        "width": aim_shape[1],
        "x1": int(center_x-width/2),
        "x2": int(center_x+width/2),
        "y1": int(center_y-height/2),
        "y2": int(center_y+height/2),
        "key_points": list(key_points)
    }


    return True, info_dict




def detect(save_img=False):

    cv2.namedWindow("test") 
    cv2.setMouseCallback("test", OnMouseAction)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    weigths = torch.load('yolov7-mask.pt')
    model_seg = weigths['model']
    model_seg = model_seg.half().to(device)
    _ = model_seg.eval()


    weigths_pose = torch.load('yolov7-w6-pose.pt')
    model_pose = weigths_pose['model']
    model_pose = model_pose.half().to(device)
    _ = model_pose.eval()


    from sma.segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)


    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    #Load the count log file
    with open(opt.logfile_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    
    img_info_list = content["images_list"]
    pic_idx = content["images_count"]
    for path, img, im0s, vid_cap in dataset:
        valid_img = False
        info_list = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                bbx_idx =0 
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if conf >= 0.85 and names[int(cls)]=="person":
                        # print(i)
                        is_valid, info_dict = save_sub_image(im0, xyxy, pic_idx, bbx_idx, device, model_seg, model_pose, predictor, hyp)
                        if is_valid:
                            info_list.append(info_dict)
                            valid_img = True
                            bbx_idx += 1
                    #####   HERE    ####

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        if valid_img:
            save_path = "dataset/used_raw_images/" + str(pic_idx).zfill(5) +".jpg"
            cv2.imwrite(save_path, im0)
            img_info = {
                "path": save_path,
                "sub_images": info_list
            }
            img_info_list.append(img_info)
            pic_idx += 1

    #Update the count log file
    new_content = {
        "images_count": pic_idx,
        "images_list": img_info_list
    }
    with open(opt.logfile_path, "w", encoding="utf-8") as f:
        json.dump(new_content, f, indent=4)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--logfile-path', type=str, default='dataset/datalog.json')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
