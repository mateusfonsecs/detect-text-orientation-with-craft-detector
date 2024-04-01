import os
import time

import torch
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
from utils import craft_utils, file_utils, imgproc

from nets.nn import CRAFT, RefineNet
import yaml
from collections import OrderedDict

trained_model= 'weights/craft_mlt_25k.pth'          # pretrained model
text_threshold= 0.7                                 # text confidence threshold
low_text= 0.4                                       # text low-bound score
link_threshold= 0.4                                 # link confidence threshold
cuda= False                                          # Use cuda for inference
canvas_size= 1280                                   # image size for inference
mag_ratio= 1.5                                      # image magnification ratio
poly= False                                         # enable polygon type
show_time= False                                    # show processing time
test_folder= 'data'                                 # folder path to input images
refine= False                                       # enable link refiner
refiner_model= 'weights/craft_refiner_CTW1500.pth'  # pretrained refiner model

with open(os.path.join('utils', 'config.yaml')) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args['test_folder'])

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def read_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args['canvas_size'],
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args['show_time']: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def read(image, text_threshold = text_threshold, link_threshold= link_threshold, low_text= low_text, cuda= cuda, poly= poly, refine_net=None):
    image_path = 'result/'
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args['trained_model'] + ')')
    if args['cuda']:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'])))
    else:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'], map_location='cpu')))

    if args['cuda']:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args['refine']:

        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args['refiner_model'] + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'])))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'], map_location='cpu')))

        refine_net.eval()
        args['poly'] = True

    t = time.time()

    # load data
    image = imgproc.loadImage(image)

    bboxes, polys, score_text = read_net(net, image, text_threshold = text_threshold, link_threshold= link_threshold, low_text= low_text, cuda= cuda, poly= poly, refine_net=None)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

if __name__ == '__main__':
    # load net
    read('/home/dg/aleatorio/ufv/detect-text-orientation-with-craft-detector/result.jpg')
