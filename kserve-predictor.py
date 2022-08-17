import argparse
import kserve
from typing import Dict
from PIL import Image
import base64
import io
import os

from utils import transforms
from PTGAN_test_for_one_CCK import make_arg_parser, parse_config_file, get_one_img
from process_for_test_CCK import get_pose, do_inference_reid

from globals import preload_pose_model, preload_reid_models


def _make_transformer(cfg):
    normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = transforms.Compose([transforms.RectScale(256, 256), transforms.ToTensor(), normalizer])
    return transform

class SP8ModelPoseAndReId(kserve.Model):
    """ SP8ModelPoseAndReId calculates pose id and reid with gpu modens
    """
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.device = "cuda"
       self.cfg = None

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
        #      "poseid": "<poseid>",
        #      "reid_feat": <b64 str>,
        #    },
        #  ]
        # }
        query_name = "0002_c002_00030600_0.jpg" # this format should be static. otherwise then parsing pid/camid would fail
        query_path = "/tmp/{}".format(query_name)
        data = inputs[0]["image_bytes"]["b64"]
        key = inputs[0]["key"]
        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data))
        input_image.save(query_path)

        transform = _make_transformer(self.cfg)
        query_data = get_one_img(query_path, transform=transform)
        pose_id_response = get_pose(query_data, self.device)
        query_feats = do_inference_reid(self.cfg, query_data, self.device)

        return {
                "predictions": [
                {
                    "poseid": pose_id_response,
                    "reid_feat": query_feats,
                    }
            ]
        }

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = SP8ModelPoseAndReId(args.model_name)
    model.load()
    kserve.ModelServer().start([model])
