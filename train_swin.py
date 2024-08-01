# LICENSE

# IMPORTS
import torch
import os
import uuid
import random
import sys
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.tempo_utils import SliWinManager
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, build_neighbor
from utils.image_utils import psnr


from scene import DynamicScene
from scene import SwinGaussianModel
from gaussian_renderer import deformable_render as render
from scene.gaussian_model import build_scaling_rotation



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# CONSTS

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
                    swin_mgr: SliWinManager):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cams = sorted(scene.batchGetTestCam(swin_mgr.all_frames()),
                           key=lambda x: x.frame)
        # train_cams = scene.batchGetTrainCamAt(swin_mgr.sampled_frames())
        validation_configs = ({'name': 'test', 'cameras' : test_cams}, 
                            #   {'name': 'train', 'cameras' : [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)]}
                              )

        grouping = lambda x: x.split('/')[0]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_test_per_frame = {}
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
                    psnr_test_per_frame.setdefault(grouping(viewpoint.image_name), []).append(psnr(image, gt_image).mean().double())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # ---------------------------- manual dump result ---------------------------- #
                with open("result.txt", "a") as f:
                    for idx, psnr_list in psnr_test_per_frame.items():
                        f.write("\n[ITER {} FRAME {}] eval {} PSNR {}".format(iteration, idx, config['name'], sum(psnr_list)/len(psnr_list)))
                
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
                       genesis: bool = False,
                       first_iter: int = 0):

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # training set setup
    # free up gpu memory 
    viewpoint_stack = None
    # not necessary, as long as you use batchGet, which is safe
    # scene.unloadAllFrames()
    
    ema_loss_for_log = 0.0
    total_iterations = train_args.iterations
    progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
    first_iter += 1

    gaussians.training_setup(train_args)

    neighbor = None

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
            viewpoint_stack = scene.batchGetTrainCam(
                swin_mgr.sampled_frames()).copy()
            random.shuffle(viewpoint_stack)
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # breakpoint()
        render_ret = render(viewpoint_cam, gaussians, pipe_args, bg)

        image = render_ret['render']
        active_pc = render_ret['input_gaussians']
        gt_image = viewpoint_cam.original_image.cuda()

        # image loss
        Ll1 = l1_loss(image, gt_image)
        Lssim = ssim(image, gt_image)
        loss = (1.0 - train_args.lambda_dssim) * Ll1 + train_args.lambda_dssim * (1.0 - Lssim)
        # mcmc regularization
        loss += args.opacity_reg * torch.abs(active_pc['opacity']).mean()
        loss += args.scale_reg * torch.abs(active_pc['scaling']).mean()
        # arap regularization
        

        loss.backward()

        # if iter > 606 and iter < 10000:
        #     print(f"iter: {iter}, image shape: {image.shape}, gt shape: {gt_image.shape}")
        #     breakpoint()

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
            # TODO, try to change different location for densification
            if iter == train_args.densify_from_iter:
                neighbor = build_neighbor(gaussians.get_immature_para(['xyz'])['xyz'])
            if iter < train_args.densify_until_iter and iter > train_args.densify_from_iter and iter % train_args.densification_interval == 0:
                gaussians.relocate_gs_immuture(swin_mgr, iter % (train_args.densification_interval * 10)== 0)
                neighbor = build_neighbor(gaussians.get_immature_para(['xyz'])['xyz'])

                if genesis: # only increasing gaussian number for genesis
                    gaussians.add_new_gs(cap_max=args.cap_max)
            
            if iter < total_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # ----------------------------- add perturbation ----------------------------- #
                # we only perturb immature gaussians

                # we only perturb the active set of immature gaussians, not all of them 
                # get immature_mask from render_ret
                immature_pc = gaussians.get_immature_para()
                immature_active_idx = gaussians.derive_idx_of_active(viewpoint_cam.frame)[0]

                L = build_scaling_rotation(immature_pc['scaling'][immature_active_idx], immature_pc['rotation'][immature_active_idx])
                noise_spread = L @ L.transpose(1, 2)
                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                noise_intensity = torch.randn_like(immature_pc['xyz'][immature_active_idx]) * \
                        ( op_sigmoid(1 - immature_pc['opacity'][immature_active_idx]) ) * \
                        args.noise_lr * xyz_lr
                noise_intensity = torch.bmm(noise_spread, noise_intensity.unsqueeze(-1)).squeeze(-1)

                #  _xyz it the para that we want to directly perturb
                gaussians._xyz[immature_active_idx].add_(noise_intensity)

            # -------------------------------- check point ------------------------------- #
            # training status save
            if (iter in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iter))
                torch.save((gaussians.capture(), swin_mgr.state_dump(), iter), f"{scene.model_path}/chkpnt_{swin_mgr.frame_start}_{iter}.pth")

def train(dataset_args, train_args, pipe_args, args):
    safe_state(args.quiet)
    print(f"Dectect anomaly: {args.detect_anomaly}")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    tb_writer = prepare_output_and_logger(dataset_args)

    # ----------------------------------- init ----------------------------------- #
    gaussians = SwinGaussianModel(dataset_args.sh_degree,
                                  max_lifespan=args.swin_size,
                                  matured_buffer_size=args.cap_max,
                                  disable_deform=args.no_deform)
    scene = DynamicScene(dataset_args, gaussians)
    swin_mgr = SliWinManager(args.swin_size,
                             scene.max_frame,
                             DynamicScene.MAX_FRAME_IN_MEMORY)

    first_iter = 0
    if args.start_checkpoint:
        print(f"Loading checkpoint {args.start_checkpoint}")
        checkpoint = torch.load(args.start_checkpoint)
        gaussians.restore(checkpoint[0], train_args)
        swin_mgr.state_load(checkpoint[1])
        first_iter = checkpoint[2]
        print(f"Checkpoint {first_iter} of {swin_mgr} loaded")
    
    # finish init window first
    genesis = swin_mgr.frame_start == 0
    train_slide_window(dataset_args, train_args, pipe_args, args,
                    gaussians, scene, swin_mgr, tb_writer, 
                    genesis=genesis,
                    first_iter=first_iter)
    if args.first_frame_only:
        return
    if genesis:
        gaussians.decay_genesis()
    swin_mgr.tick()

    # ----------------------- start sliding window training ---------------------- #
    while swin_mgr.frame_end <= swin_mgr.max_frame:
    # while swin_mgr.frame_start < swin_mgr.max_frame:
        # any current immature gaussian whose 
        #   life ends strictly before window_end, or to say
        #   can not fulfill the whole window range, will:
        # 1. reproduce a new gaussian
        # 2. get matured and moved to matured list
        gaussians.evolve(swin_mgr) 

        train_slide_window(dataset_args, train_args, pipe_args, args, 
                           gaussians, scene, swin_mgr, tb_writer)

        print(f"retiring frame #{swin_mgr.frame_start}")
        swin_mgr.tick()

    # there are some immature gaussians in the last frame
    # mature them
    gaussians.mature_rest()

def parse():
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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[29_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--swin_size", type=int, default=5)
    parser.add_argument("--first_frame_only", action="store_true", default=False)
    parser.add_argument("--no_deform", action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    dataset_args = lp.extract(args)
    train_args = op.extract(args)
    pipe_args = pp.extract(args)

    return dataset_args, train_args, pipe_args, args

if __name__ == "__main__":
    # breakpoint()
    # current version
    # ignore gaussian load/unload
    # i.e. all guassian still stays in GDDR
    # ignore check point
    random.seed(314159)
    dataset_args, train_args, pipe_args, args = parse()
    train(dataset_args, train_args, pipe_args, args)
    