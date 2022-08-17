import argparse
import kserve
from typing import Dict
from PIL import Image
import base64
import io
import os
import json
import time
import logging
import sys

from utils import transforms
from PTGAN_test_for_one_CCK import make_arg_parser, parse_config_file, get_one_img
from load_distmat import draw_ranked_photo
from process_for_test_CCK import do_inference

def _make_transformer(cfg):
    normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = transforms.Compose([transforms.RectScale(256, 256), transforms.ToTensor(), normalizer])
    return transform

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"

class SP8ModelRanking(kserve.Model):
    """ SP8ModelRankingDist calculates the ranking distances from the given
    feat and pose id with feats in gallery. This step is cpu intensive"""

    def __init__(self, name: str, predictor_host: str):
       super().__init__(name)
       self.name = name
       self.cfg = None
       self.device = "cuda"
       self.predictor_host = predictor_host

       self.max_request_count = 2
       self.current_accumulated_request_count = 0

    def load(self):

        parser = make_arg_parser()
        argmap = parser.parse_args(["--config_file", "config/stage2/transreid_256_veri_gan.yml"])
        cfg = parse_config_file(argmap)
        print("parse config\n")
        self.cfg = cfg
 
        self.ready = True
        return self.ready

    async def preprocess(self, request: Dict) -> Dict:
        logging.info("preprocessing...")
        self._self_terminated_if_necessary()

        response = await self._http_client.fetch(
            PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name),
            method='POST',
            request_timeout=self.timeout,
            body=json.dumps(request)
        )
        if response.code != 200:
            raise tornado.web.HTTPError(
                status_code=response.code,
                reason=response.body)
        prediction_response = json.loads(response.body)
        request["predictions"] = prediction_response["predictions"]
        return request

    def _self_terminated_if_necessary(self):
        if self.current_accumulated_request_count > self.max_request_count:
            logging.info("self-terminated. accumulated: {0}, maximum: {1}".format(self.current_accumulated_request_count, self.max_request_count))
            sys.exit(0)

    def _query_ts(self) -> int:
        return int(time.time())

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
        #   "predictions": [
        #     {
        #       "poseid":"pid",
        #       "reid_feat": <b64string>,
        #     }
        #   ]
        # }
        predictions = request["predictions"]
        if not predictions:
            return {
                "status": "failure",
                "message": "missing required fields"
            }
        query_feats = request["predictions"][0]["reid_feat"]
        query_poseid = request["predictions"][0]["poseid"]

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

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
parser.add_argument('--predictor_host',
                    help='The URL for the model predict functioni.', required=False)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = SP8ModelRanking(args.model_name, predictor_host=args.predictor_host)
    model.load()
    kserve.ModelServer().start([model])
