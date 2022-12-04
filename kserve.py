import argparse
import kserve
from typing import Dict
from PIL import Image
import base64
import io
import os
import time
import logging
import sys

from utils import transforms
from PTGAN_test_for_one_CCK import make_arg_parser, parse_config_file, get_one_img
from process_for_test_CCK import get_pose, do_inference_reid, do_inference
from load_distmat import draw_ranked_photo

from globals import preload_pose_model, preload_reid_models

def _make_transformer(cfg):
    normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = transforms.Compose([transforms.RectScale(256, 256), transforms.ToTensor(), normalizer])
    return transform

class SP8ModelIntegrated(kserve.Model):
    """ SP8ModelIntegrated calculates reid cars with gpu modens
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.device = "cuda"
        self.cfg = None

        # current_accumulated_request_count records the current accumuated
        # request count
        # and max_request_count is the maximum request count allowed during a
        # single live cycle
        # once $current_accumulated_request_count exceeds the maximum count
        # it will signal the program to terminate itself
        self.max_request_count = 2
        self.current_accumulated_request_count = 0

    def load(self):
        print("preload models...\n")
        preload_pose_model(self.device)

        parser = make_arg_parser()
        argmap = parser.parse_args(["--config_file", "config/stage2/transreid_256_veri_gan.yml"])

        cfg = parse_config_file(argmap)
        print("preload reid_models...\n")
        preload_reid_models(cfg, self.device)

        self.cfg = cfg
        self.ready = True
        return self.ready

    def preprocess(self, request: Dict) -> Dict:
        self._self_terminated_if_necessary()
        return request

    def _self_terminated_if_necessary(self):
        if self.current_accumulated_request_count > self.max_request_count:
            logging.info("self-terminated. accumulated: {0}, maximum: {1}".format(self.current_accumulated_request_count, self.max_request_count))
            sys.exit(0)

    def postprocess(self, response: Dict) -> Dict:
        self.current_accumulated_request_count = self.current_accumulated_request_count + 1
        if self.current_accumulated_request_count > self.max_request_count:
            logging.info("accumuated request: {0}, maximum: {1}, self-terminated flagged".format(self.current_accumulated_request_count, self.max_request_count))
        logging.info("accumuated request: {0}, maximum: {1}".format(self.current_accumulated_request_count, self.max_request_count))
        return response

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {
        #   "instances": [
        #     {
        #       "image_bytes": {
        #           "b64": "<b64-encoded>",
        #       },
        #       "key": "somekeys",
        #     },
        #   ],
        # }
        # and response is wrapped into the following
        # {
        #  "predictions: [
        #    {
        #      "ranked_image_bytes": {"b64": "<b64>"},
        #      "key": "$key",
        #      "type": "carReid",
        #    },
        #  ]
        # }
        ts = self._query_ts()
        query_name = "0002_c002_00030600_0.jpg" # this format should be static. otherwise then parsing pid/camid would fail
        query_path = "/tmp/{}".format(query_name)
        query_result_path = "/tmp/{}.result.png".format(ts)
        data = inputs[0]["image_bytes"]["b64"]
        key = inputs[0]["key"]
        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data))
        input_image.save(query_path)

        transform = _make_transformer(self.cfg)
        query_data = get_one_img(query_path, transform=transform)

        query_poseid= get_pose(query_data, self.device)

        query_feats = do_inference_reid(self.cfg, query_data, self.device)

        do_inference(self.cfg, query_data, query_feats, query_poseid, self.device)

        draw_ranked_photo("similiar_img_distmat.csv", query_result_path, 16)
        grid_result_img = Image.open(query_result_path)
        rgb_im = grid_result_img.convert('RGB')

        buffered = io.BytesIO()
        rgb_im.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {"predictions": [{
            "ranked_image_bytes": {
                "b64": img_str,
            },
            "key": key,
            "type": "carReid",
        }]}

    def _query_ts(self) -> int:
        return int(time.time())

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = SP8ModelIntegrated(args.model_name)
    model.load()
    kserve.ModelServer().start([model])
