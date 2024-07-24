# LICENSE

# IMPORTS
import torch
import os
import uuid
import sys
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.tempo_utils import SliWinManager
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


from scene import DynamicScene
from scene import GaussianModel as SwinGaussianModel
from gaussian_renderer import deformable_render as render
from scene.gaussian_model import build_scaling_rotation



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# CONSTS
SWIN_SIZE = 4 # actual is the max size 4090 could load the dataset

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: DynamicScene, renderFunc, renderArgs, args,
                    swin_mgr):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cams = swin_mgr.fetch_cams(scene.getTestCamerasAt)
        train_cams = swin_mgr.fetch_cams(scene.getTrainCamerasAt)
        validation_configs = ({'name': 'test', 'cameras' : test_cams}, 
                              {'name': 'train', 'cameras' : [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # no need to append swin_mgr to render when test time, it can just render by current frame info
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            pc = scene.gaussians.get_immature_para(['xyz', 'opacity'])
            tb_writer.add_histogram("scene/opacity_histogram", pc['opacity'], iteration)
            tb_writer.add_scalar('total_points', pc['xyz'].shape[0], iteration)
        
        torch.cuda.empty_cache()

def train_slide_window(dataset_args, train_args, pipe_args, args,
                       gaussians: SwinGaussianModel, scene: DynamicScene,
                       swin_mgr: SliWinManager, 
                       tb_writer,
                       genesis: bool = False):
    first_iter = 0

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # training set setup
    # free up gpu memory 
    viewpoint_stack = None
    scene.clearAllTrain()
    scene.clearAllTest()
    
    ema_loss_for_log = 0.0
    total_iterations = train_args.iterations if not genesis else train_args.genesis_iterations
    progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
    first_iter += 1

    gaussians.training_setup(train_args)

    for iter in range(first_iter, total_iterations):

        iter_start.record()
        xyz_lr = gaussians.update_learning_rate(iter)
        if genesis and iter % 1000 == 0:
            gaussians.oneupSHdegree()
        if (iter - 1) == args.debug_from:
            pipe_args.debug = True
        bg = torch.rand((3), device="cuda") if train_args.random_background else background

        # ------------------------------ normal training ----------------------------- #
        if not viewpoint_stack:
            viewpoint_stack = swin_mgr.fetch_cams(scene.getTrainCamerasAt)
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_ret = render(viewpoint_cam, gaussians, pipe_args, bg)

        image = render_ret['render']
        active_pc = render_ret['input_gaussians']
        immature_pc = gaussians.get_immature_para()
        gt_image = viewpoint_cam.original_image.cuda()

        # image loss
        Ll1 = l1_loss(image, gt_image)
        Lssim = ssim(image, gt_image)
        loss = (1.0 - train_args.lambda_dssim) * Ll1 + train_args.lambda_dssim * (1.0 - Lssim)
        # mcmc regularization
        loss += args.opacity_reg * torch.abs(active_pc['opacity']).mean()
        loss += args.scale_reg * torch.abs(active_pc['scaling']).mean()
        # arap regularization
        # TODO: add arap regularization

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iter == total_iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            args.test_iterations, 
                            scene, render, (pipe_args, background), args,
                            swin_mgr)
            # guassian point cloud save 
            if (iter in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iter))
                scene.save(iter)

            # ------------------------------- densification ------------------------------ #
            if iter < train_args.densify_until_iter and iter > train_args.densify_from_iter and iter % train_args.densification_interval == 0:
                gaussians.relocate_gs_immuture(swin_mgr)

                if genesis: # only increasing gaussian number for genesis
                    gaussians.add_new_gs(cap_max=args.cap_max)
            
            if iter < total_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # ----------------------------- add perturbation ----------------------------- #
                # we only perturb immature gaussians
                # TODO we only perturb the active set of immature gaussians, not all of them 
                L = build_scaling_rotation(immature_pc['scaling'], immature_pc['rotation'])
                noise_spread = L @ L.transpose(1, 2)
                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                noise_intensity = torch.randn_like(immature_pc['xyz']) * \
                        ( op_sigmoid(1 - immature_pc['opacity']) ) * \
                        args.noise_lr * xyz_lr
                noise_intensity = torch.bmm(noise_spread, noise_intensity.unsqueeze(-1)).squeeze(-1)

                #  _xyz it the para that we want to directly perturb
                gaussians._xyz.add_(noise_intensity)

            # -------------------------------- check point ------------------------------- #
            # training status save
            if (iter in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iter))
                torch.save((gaussians.capture(), iter), scene.model_path + "/chkpnt" + str(iter) + ".pth")

def train():
    # ------------------------------- args parsing ------------------------------- #
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(1_000, 30_000, 1_000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(1_000, 30_000, 5_000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    dataset_args = lp.extract(args)
    train_args = op.extract(args)
    pipe_args = pp.extract(args)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    tb_writer = prepare_output_and_logger(dataset_args)

    # ----------------------------------- init ----------------------------------- #
    gaussians = SwinGaussianModel(dataset_args.sh_degree, max_lifespan=SWIN_SIZE)
    scene = DynamicScene(dataset_args, gaussians)
    swin_mgr = SliWinManager(SWIN_SIZE, scene.max_frame)

    # ---------------------------------- bootup ---------------------------------- #
    # swin_mgr as [0, N)
    # train with all MAX_CAP gs with life [0, N)
    # call then genesis gs
    train_slide_window(dataset_args, train_args, pipe_args, args, 
                       gaussians, scene, swin_mgr, tb_writer, True)

    # --------------------------- decay the genesis gs: -------------------------- #
    # each 1/N's life was reduced to [0, i) i in [1..N]
    gaussians.decay_genesis() # genesis is not mature yet

    swin_mgr.tick() # move the window to [1, N+1)
    print(f"retiring frame #{swin_mgr.frame_start}")
    scene.clearTrainCamerasAt(swin_mgr.frame_start) # retire the frame
    scene.clearTestCamerasAt(swin_mgr.frame_start) # retire the frame

    # any current immature gaussian whose 
    #   life ends strictly before window_end, or to say
    #   can not fulfill the whole window range, will:
    # 1. reproduce a new gaussian
    # 2. get matured and moved to matured list
    gaussians.evolve(swin_mgr) 

    # ----------------------- start sliding window training ---------------------- #
    while swin_mgr.frame_end <= swin_mgr.max_frame:
        train_slide_window(dataset_args, train_args, pipe_args, args, 
                           gaussians, scene, swin_mgr, tb_writer)
        
        if swin_mgr.frame_end < swin_mgr.max_frame:
            swin_mgr.tick()
            gaussians.evolve(swin_mgr)

if __name__ == "__main__":
    # current version
    # ignore gaussian load/unload
    # i.e. all guassian still stays in GDDR
    # ignore check point
    train()
    