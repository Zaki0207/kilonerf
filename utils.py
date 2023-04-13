import os
import signal
import time
from collections import deque, defaultdict
from itertools import product
import numpy as np
import argparse
import yaml
import run_nerf_helpers
import torch
from tqdm import tqdm
import lpips
# from kornia import create_meshgrid

# BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
#                                device='cuda')

def get_random_points_inside_domain(num_points, domain_min, domain_max):
    x = np.random.uniform(domain_min[0], domain_max[0], size=(num_points,))
    y = np.random.uniform(domain_min[1], domain_max[1], size=(num_points,))
    z = np.random.uniform(domain_min[2], domain_max[2], size=(num_points,))
    return np.column_stack((x, y, z))
    
def get_random_directions(num_samples):
    random_directions = np.random.randn(num_samples, 3)
    random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1, 1)
    return random_directions

def load_pretrained_nerf_model(dev, cfg):
    pretrained_cfg = load_yaml_as_dict(cfg['pretrained_cfg_path'])
    pretrained_cfg['bounding_box'] = cfg['bounding_box']
    if 'use_initialization_fix' not in pretrained_cfg:
        pretrained_cfg['use_initialization_fix'] = False
    if 'num_importance_samples_per_ray' not in pretrained_cfg:
        pretrained_cfg['num_importance_samples_per_ray'] = 0
    pretrained_nerf, embed_fn, embeddirs_fn = create_nerf(pretrained_cfg)
    pretrained_nerf = pretrained_nerf.to(dev)
    checkpoint = torch.load(cfg['pretrained_checkpoint_path'])
    pretrained_nerf.load_state_dict(checkpoint['model_state_dict'])
    if pretrained_cfg['turn_on_hash_feature']:
        embed_fn.load_state_dict(checkpoint['embed_fn_state_dict'])
        embed_fn = embed_fn.cuda()
    pretrained_nerf = run_nerf_helpers.ChainEmbeddingAndModel(pretrained_nerf, embed_fn, embeddirs_fn) # pos. encoding
    return pretrained_nerf

def create_nerf(cfg):
    
    i_embed = 0 # 使用傅立叶变换特征：1
    if cfg['turn_on_hash_feature']: # 使用哈希编码：0
        i_embed = 1
        
    embed_fn, input_ch = run_nerf_helpers.get_embedder(cfg['num_frequencies'], cfg, i_embed)
    
    if i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())
        
    input_ch_views = 0
    embeddirs_fn = None
    
    if cfg['use_viewdirs']:
        embeddirs_fn, input_ch_views = run_nerf_helpers.get_embedder(cfg['num_frequencies_direction'], cfg, 2)
    else:
        embeddirs_fn, input_ch_views = run_nerf_helpers.get_embedder(cfg['num_frequencies_direction'], cfg, 0)
    output_ch = 5 if cfg['num_importance_samples_per_ray'] > 0 else 4
    skips = [cfg['refeed_position_index']]
    
    if i_embed==1:
        model = run_nerf_helpers.NeRFSmall(num_layers=cfg['hash_mode_num_layers'],
                        hidden_dim=cfg['hash_mode_hidden_dim'],
                        geo_feat_dim=cfg['hash_mode_geo_feat_dim'],
                        num_layers_color=cfg['hash_mode_num_layers_color'],
                        hidden_dim_color=cfg['hash_mode_hidden_dim_color'],
                        input_ch=input_ch, input_ch_views=input_ch_views)
    else:
        model = run_nerf_helpers.NeRF(D=cfg['num_hidden_layers'], W=cfg['hidden_layer_size'],
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=True,
                    direction_layer_size=cfg['direction_layer_size'], use_initialization_fix=cfg['use_initialization_fix'])
    
    if cfg['num_importance_samples_per_ray'] > 0:
        if i_embed==1:
            model_fine = run_nerf_helpers.NeRFSmall(num_layers=cfg['hash_mode_num_layers'],
                                            hidden_dim=cfg['hash_mode_hidden_dim'],
                                            geo_feat_dim=cfg['hash_mode_geo_feat_dim'],
                                            num_layers_color=cfg['hash_mode_num_layers_color'],
                                            hidden_dim_color=cfg['hash_mode_hidden_dim_color'],
                                            input_ch=input_ch, input_ch_views=input_ch_views)
        
        else:
            model_fine = run_nerf_helpers.NeRF(D=cfg['num_hidden_layers'], W=cfg['hidden_layer_size'],
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=True,
                            direction_layer_size=cfg['direction_layer_size'], use_initialization_fix=cfg['use_initialization_fix'])
        model = run_nerf_helpers.CoarseAndFine(model, model_fine)
    
    return model, embed_fn, embeddirs_fn    
    
def query_densities(points, pretrained_nerf, cfg, dev):
    mock_directions = torch.zeros_like(points) # density does not depend on direction
    points_and_dirs = torch.cat([points, mock_directions], dim=1)
    num_points_and_dirs = points_and_dirs.size(0)
    densities = torch.empty(num_points_and_dirs)
    if 'query_batch_size' in cfg:
        query_batch_size = cfg['query_batch_size']
    else:
        query_batch_size = num_points_and_dirs
    with torch.no_grad():
        start = 0
        while start < num_points_and_dirs:
            end = min(start + query_batch_size, num_points_and_dirs)
            densities[start:end] = F.relu(pretrained_nerf(points_and_dirs[start:end].to(dev))[:, -1]).cpu() # Only select the densities (A) from NeRF's RGBA output
            start = end
    return densities
    
def has_flag(cfg, name):
    return name in cfg and cfg[name]
    
def load_yaml_as_dict(path):
    with open(path) as yaml_file:
        yaml_as_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_as_dict
    
def parse_args_and_init_logger(default_cfg_path=None, parse_render_cfg_path=False):
    parser = argparse.ArgumentParser(description='NeRF distillation')
    parser.add_argument('cfg_path', type=str)
    parser.add_argument('log_path', type=str, nargs='?')
    if parse_render_cfg_path:
        parser.add_argument('-rcfg', '--render_cfg_path', type=str)
    args = parser.parse_args()
    cfg = load_yaml_as_dict(args.cfg_path)
    if args.log_path is None:
        start = args.cfg_path.find('/')
        end = args.cfg_path.rfind('.')
        args.log_path = 'logs' + args.cfg_path[start:end]
        if cfg['turn_on_hash_feature']:
            args.log_path += '_hash'
            args.log_path += str(cfg['n_hash_feature'])
        print('auto log path:', args.log_path)
    
    create_directory_if_not_exists(args.log_path)
    Logger.filename = args.log_path + '/log.txt'
    
    # cfg = load_yaml_as_dict(args.cfg_path)
    if default_cfg_path is not None:
        default_cfg = load_yaml_as_dict(default_cfg_path)
        for key in default_cfg:
            if not key in cfg:
                cfg[key] = default_cfg[key]
    print(cfg)
    
    ret_val = (cfg, args.log_path)
    if parse_render_cfg_path:
        ret_val += (args.render_cfg_path,)
        
    return ret_val
    
class IterativeMean:
    def __init__(self):
        self.value = None
        self.num_old_values = 0
    
    def add_values(self, new_values):
        if self.value:
            self.value = (self.num_old_values * self.value + new_values.size(0) * new_values.mean()) / (self.num_old_values + new_values.size(0))
        else:
            self.value = new_values.mean()
        self.num_old_values += new_values.size(0)
        
    def get_mean(self):
        return self.value.item()


def create_directory_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        
class Logger:
    filename = None
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            print(text, flush=True)
            log_file.write(text + '\n')

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGUSR1, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        
def extract_domain_boxes_from_tree(root_node):
    nodes_to_process = deque([root_node])
    boxes = []
    while nodes_to_process:
        node = nodes_to_process.popleft()
        if hasattr(node, 'leq_child'):
            nodes_to_process.append(node.leq_child)
            nodes_to_process.append(node.gt_child)
        else:
            boxes.append([node.domain_min, node.domain_max])
    
    return boxes

def write_boxes_to_obj(boxes, obj_filename):
    txt = ''
    i = 0
    for box in tqdm(boxes):
        for min_or_max in product(range(2), repeat=3):
            txt += 'v {} {} {}\n'.format(box[min_or_max[0]][0], box[min_or_max[1]][1], box[min_or_max[2]][2])
        for x, y, z in [(0b000, 0b100, 0b010), (0b100, 0b010, 0b110),
            (0b001, 0b101, 0b011), (0b101, 0b011, 0b111),
            (0b000, 0b010, 0b001), (0b001, 0b011, 0b010),
            (0b100, 0b110, 0b101), (0b101, 0b111, 0b110),
            (0b000, 0b100, 0b001), (0b100, 0b101, 0b001),
            (0b010, 0b110, 0b011), (0b110, 0b111, 0b011)]:
            txt += 'f {} {} {}\n'.format(1 + i * 8 + x, 1 + i * 8 + y, 1 + i * 8 + z)
        i += 1
    
    with open(obj_filename, 'a') as obj_file:
        obj_file.write(txt)

class PerfMonitor:
    events = []
    is_active = True

    @staticmethod
    def add(name, groups=[]):
        if PerfMonitor.is_active:
            torch.cuda.synchronize()
            t = time.perf_counter()
            PerfMonitor.events.append((name, t, groups))
    
    @staticmethod
    def log_and_reset(write_detailed_log):
        previous_t = PerfMonitor.events[0][1]
        group_map = defaultdict(float)
        elapsed_times = []
        for name, t, groups in PerfMonitor.events[1:]:
            elapsed_time = t - previous_t
            elapsed_times.append(elapsed_time)
            for group in groups:
                group_map[group] += elapsed_time
            group_map['total'] += elapsed_time
            previous_t = t
        max_length = max([len(name) for name, _, _ in PerfMonitor.events] + [len(group) for group in group_map])
        
        if write_detailed_log:
            for event, elapsed_time in zip(PerfMonitor.events[1:], elapsed_times):
                name = event[0]
                extra_whitespace = ' ' * (max_length - len(name))
                Logger.write('{}:{} {:7.2f} ms'.format(name, extra_whitespace, 1000 * (elapsed_time)))
            Logger.write('')
            for group in group_map:
                extra_whitespace = ' ' * (max_length - len(group))
                Logger.write('{}:{} {:7.2f} ms'.format(group, extra_whitespace, 1000 * (group_map[group])))
        
        # Reset        
        PerfMonitor.events = []
        
        return group_map['total']

class LPIPS:
    loss_fn_alex = None 
    
    @staticmethod
    def calculate(img_a, img_b):
        img_a, img_b = [img.permute([2, 1, 0]).unsqueeze(0) for img in [img_a, img_b]]
        if LPIPS.loss_fn_alex == None: # lazy init
            LPIPS.loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')
        return LPIPS.loss_fn_alex(img_a, img_b)
    
    
def get_distance_to_closest_point_in_box(point, domain_min, domain_max):
    closest_point = np.array([0., 0., 0.])
    for dim in range(3):
        if point[dim] < domain_min[dim]:
            closest_point[dim] = domain_min[dim]
        elif domain_max[dim] < point[dim]:
            closest_point[dim] = domain_max[dim]
        else: # in between domain_min and domain_max
            closest_point[dim] = point[dim]
    return np.linalg.norm(point - closest_point)
    
    
def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    
    
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    dir_bounds = directions.view(-1, 3)
    # print("Directions ", directions[0,0,:], directions[H-1,0,:], directions[0,W-1,:], directions[H-1, W-1, :])
    # print("Directions ", dir_bounds[0], dir_bounds[W-1], dir_bounds[H*W-W], dir_bounds[H*W-1])

    return directions    
    
def get_distance_to_furthest_point_in_box(point, domain_min, domain_max):
    furthest_point = np.array([0., 0., 0.])
    for dim in range(3):
        mid = (domain_min[dim] + domain_max[dim]) / 2
        if point[dim] > mid:
            furthest_point[dim] = domain_min[dim]
        else:
            furthest_point[dim] = domain_max[dim]
    return np.linalg.norm(point - furthest_point)
   
def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)   
    
class ConfigManager:
    global_domain_min = None
    global_domain_max = None

    @staticmethod
    def init(cfg):
        if 'global_domain_min' in cfg and 'global_domain_max' in cfg:
            ConfigManager.global_domain_min = cfg['global_domain_min']
            ConfigManager.global_domain_max = cfg['global_domain_max']
        elif 'dataset_dir' in cfg and cfg['dataset_type'] == 'nsvf':
            bbox_path = os.path.join(cfg['dataset_dir'], 'bbox.txt')
            bounding_box = load_matrix(bbox_path)[0, :-1]
            ConfigManager.global_domain_min = bounding_box[:3]
            ConfigManager.global_domain_max = bounding_box[3:]
     
    @staticmethod  
    def get_global_domain_min_and_max(device=None):
        result = ConfigManager.global_domain_min, ConfigManager.global_domain_max
        if device:
            result = [torch.tensor(x, dtype=torch.float, device=device) for x in result]
        return result

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_bbox3d_for_blenderobj(poses, intrinsics, near=2.0, far=6.0):
    
    focal = intrinsics.fx
    H = intrinsics.H
    W = intrinsics.W
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for pose in poses:
        c2w = torch.FloatTensor(pose)
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

# def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
#     '''
#     xyz: 3D coordinates of samples. B x 3
#     bounding_box: min and max x,y,z coordinates of object bbox
#     resolution: number of voxels per axis
#     '''
#     box_min, box_max = bounding_box

#     keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
#     if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
#         # print("ALERT: some points are outside bounding box. Clipping them!")
#         # xyz = torch.clamp(xyz, min=box_min, max=box_max)
        
#         xyz = torch.stack([torch.clamp(xyz[:,0],min=box_min[0],max=box_max[0]),torch.clamp(xyz[:,1],min=box_min[1],max=box_max[1]),
#                    torch.clamp(xyz[:,2],min=box_min[2],max=box_max[2])],dim=1)

#     grid_size = (box_max-box_min)/resolution
    
#     bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
#     voxel_min_vertex = bottom_left_idx*grid_size + box_min
#     voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

#     voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
#     hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

#     return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

        
def main():
    if False:
        boxes = [
            [[-0.078125, 0.390625, 0.546875], [-0.0625, 0.40625, 0.5625]],
            [[-0.625, -0.375, -0.375], [-0.5, -0.25, -0.25]], 
            [[-0.625, -0.25, -0.375], [-0.5, -0.125, -0.25]],
            [[-0.625, -0.125, -0.375], [-0.5, 0.0, -0.25]],
            [[-0.5, 0.5, 0.0], [-0.375, 0.625, 0.25]],
            [[-0.125, 0.125, -0.5], [0.0, 0.25, -0.25]],
            [[-0.375, -0.25, 0.0], [-0.25, -0.125, 0.25]],
            [[-0.625, -0.5, -0.5], [-0.5, -0.375, -0.25]],
            [[-0.125, 0.0, 0.5], [0.0, 0.25, 0.75]]
        ]
    boxes = [[[0.15625, -0.3125, 0.8125], [0.1875, -0.25, 0.875]]]
    print(boxes)
    write_boxes_to_obj(boxes, 'hard_domains_2.obj')

if __name__ == '__main__':
    main()
