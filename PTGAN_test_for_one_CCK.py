import warnings

warnings.filterwarnings("ignore")
import os
import torch
import argparse
from PIL import Image
from config import cfg

torch.multiprocessing.set_sharing_strategy('file_system')

def get_one_img(query_img_path, transform):
    img = Image.open(query_img_path).convert("RGB")
    img = transform(img)
    pid = int(query_img_path[-24:-20])
    camid = int(query_img_path[-18:-15])

    return {'origin': img,
            'pid': pid,
            'camid': camid,
            'trackid': -1,
            'file_name': query_img_path
            }

 
def parse_config_file(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


#def setup_and_run(args):
#    cfg = parse_config_file(args)
#    #if args.config_file != "":
#    #    cfg.merge_from_file(args.config_file)
#    #cfg.merge_from_list(args.opts)
#    #cfg.freeze()
#    #os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
#    output_dir = cfg.OUTPUT_DIR
#    if output_dir and not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    logger = setup_logger("reid_baseline", output_dir, if_train=False)
#    logger.info(args)
#
#    #if args.config_file != "":
#    #    logger.info("Loaded configuration file {}".format(args.config_file))
#    #    with open(args.config_file, 'r') as cf:
#    #        config_str = "\n" + cf.read()
#    #        logger.info(config_str)
#    #logger.info("Running with config:\n{}".format(cfg))
#
#    normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#
#    transform = transforms.Compose([transforms.RectScale(256, 256),
#                                    transforms.ToTensor(),
#                                    normalizer])
#
#    query_data = get_one_img('../AIC21/veri_pose/query/0002_c002_00030600_0.jpg', transform=transform)
#    query_poseid = get_pose(query_data)
#    query_feats = do_inference_reid(cfg, query_data)
#    do_inference(cfg, query_data, query_feats=query_feats, query_poseid=query_poseid)

    # model = Model()
    # model.reset_model_status()
    # do_inference(cfg, query_data)

def make_arg_parser():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--query_path", default="", help="path for query photo", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser
#
#def main():
#    parser = make_arg_parser()
#    args = parser.parse_args()
##    setup_and_run(args)
#
#if __name__ == '__main__':
#    main()
