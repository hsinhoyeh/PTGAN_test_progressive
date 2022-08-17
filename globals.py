from gan.model import Model
from reid_model.make_model import make_model

pose_model = {}
def preload_pose_model(device):
    global pose_model
    pose_model= Model(device)
    pose_model.reset_model_status()
    pose_model.eval()


reid_model = {}
def preload_reid_models(cfg, device):
    global reid_model
    reid_model = make_model(cfg, num_class=1678)
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.to(device)
    reid_model.eval()
