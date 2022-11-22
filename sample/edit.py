# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
import random


MOTION_LENGTH_INDEX = 3
NUMBER_OF_TAKEN_FRAMES = 40
NUMBER_OF_CUT_FRAMES = 40
SEQUENCE_LENGTH = 5


def set_args(input, model_kwargs, frame_colors, shape, device, number_of_taken_frames, max_frames, transition_color="blue"):
    new_input = input[:, :, :, -number_of_taken_frames:]

    # pad the motions to the desired shape
    padding_shape = [(0, 0), (0, 0), (0, 0), (0, shape[3] - number_of_taken_frames)]
    new_input = torch.from_numpy(np.pad(new_input.cpu().numpy(), padding_shape))

    new_input = new_input.to(device=device)

    # set motions as argument
    model_kwargs["y"]["inpainted_motion"] = new_input

    # update the lengths argument to generate the full range of motion
    model_kwargs["y"]["lengths"] = torch.full((len(new_input),), new_input.shape[MOTION_LENGTH_INDEX] - NUMBER_OF_CUT_FRAMES, device=device)

    # set the inpainting mask
    model_kwargs["y"]["inpainting_mask"] = torch.zeros_like(new_input, dtype=torch.float, device=device)

    for i, length in enumerate(model_kwargs["y"]["lengths"].cpu().numpy()):
        # add the frame colors
        curr_number_of_taken_frames = min(number_of_taken_frames, length)
        new_frame_colors = [transition_color] * curr_number_of_taken_frames + ["orange"] * (length - curr_number_of_taken_frames)
        frame_colors[i] = frame_colors.get(i, [])[:-number_of_taken_frames] + new_frame_colors

        # set a linear mask in the transition part and then a regular mask for the inpainting part
        # linear_mask = torch.linspace(start=1, end=0, steps=curr_number_of_taken_frames)
        linear_mask_short_end = torch.cat((torch.ones(curr_number_of_taken_frames // 2), torch.linspace(start=1, end=0, steps=curr_number_of_taken_frames // 2)))
        # fixed_mask = torch.ones(curr_number_of_taken_frames)

        model_kwargs["y"]["inpainting_mask"][i, :, :, :curr_number_of_taken_frames] = linear_mask_short_end


def set_text_args(args, model_kwargs, segment_i):
    possible_texts = [
        # "A person is walking in a circle",
        # "A person is swinging a sword",
        # "A person is squatting",
        # "A person is walking backwards",
        # "A person is dancing",
        # "A person is playing basketball",
        "A person keeps running",
        # "A person is walking sideways",
    ]

    # text = random.choice(possible_texts)
    text = possible_texts[segment_i % len(possible_texts)]

    print(f"text condition: {text}")

    texts = [text] * args.num_samples
    model_kwargs["y"]["text"] = texts
    # if args.text_condition == '':
    #     args.guidance_param = 0.  # Force unconditioned generation


def main():
    # handle seed
    args = edit_args()
    fixseed(args.seed)
    random.seed(args.seed)

    # handling dataset
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    dist_util.setup_dist(args.device)
    if out_path == "":
        out_path = os.path.join(os.path.dirname(args.model_path), "edit_{}_{}_{}_seed{}".format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != "":
            out_path += "_" + args.text_condition.replace(" ", "_").replace(".", "")

    print("Loading dataset...")
    assert args.num_samples <= args.batch_size, f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split="test", hml_mode="train")  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    frame_colors = {}

    # set_args(input_motions, model_kwargs, frame_colors, input_motions.shape, input_motions.device, NUMBER_OF_TAKEN_FRAMES, max_frames)

    # model_kwargs['y']['inpainted_motion'] = input_motions
    # if args.edit_mode == 'in_between':
    #     model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
    #                                                            device=input_motions.device)  # True means use gt motion
    #     for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):

    #         # added by roy
    #         # start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
    #         start_idx, end_idx = int(args.prefix_end * length), int(length)

    #         gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
    #         model_kwargs['y']['inpainting_mask'][i, :, :,
    #         start_idx: end_idx] = False  # do inpainting in those frames
    # =================================== Fix Me =========================================
    # this section is not handles inside set_args
    # elif args.edit_mode == 'upper_body':
    #     model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
    #                                                         device=input_motions.device)  # True is lower body data
    #     model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
    #         -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

    all_motions = []
    all_lengths = []
    all_text = [""] * args.num_samples

    # add CFG scale to batch
    model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    sample_fn = diffusion.p_sample_loop

    for rep_i in range(args.num_repetitions):
        print(f"### Start sampling [repetitions #{rep_i}]")

        total_motion = input_motions[:, :, :, :NUMBER_OF_TAKEN_FRAMES]
        # total_motion = torch.Tensor(input_motions.shape[:-1] + (0,))

        curr_color = "blue"

        # every motion is comprised of a sequence of SEQUENCE_LENGTH segments
        for segment_i in range(SEQUENCE_LENGTH):

            set_args(total_motion, model_kwargs, frame_colors, input_motions.shape, input_motions.device, NUMBER_OF_TAKEN_FRAMES, max_frames, curr_color)
            set_text_args(args, model_kwargs, segment_i)

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample[:, :, :, :-NUMBER_OF_CUT_FRAMES]
            # sample = torch.zeros(input_motions.shape, device=input_motions.device)
            # ========================================== Fix Me =========================================================
            #  need to take the motions from the part that was generated, which is not necessarily the length of the vector
            print(f"before total_motion.shape {total_motion.shape}")

            total_motion = torch.cat((total_motion[:, :, :, :-NUMBER_OF_TAKEN_FRAMES], sample), dim=MOTION_LENGTH_INDEX)

            curr_color = "purple"

            all_text = [txt1 + ";" + txt2 for txt1, txt2 in zip(all_text, model_kwargs["y"]["text"])]

        # ================================= end of sequence generation =================================
        sample = total_motion

        print(f"final sample.shape {sample.shape}")

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == "hml_vec":
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_motions.append(sample.cpu().numpy())

        print(f"sample.shape: {sample.shape}")

        all_lengths.append(np.full((sample.shape[0],), sample.shape[MOTION_LENGTH_INDEX]))

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {"motion": all_motions, "text": all_text, "lengths": all_lengths, "num_samples": args.num_samples, "num_repetitions": args.num_repetitions})
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == "kit" else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == "hml_vec":
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    for sample_i in range(args.num_samples):
        caption = "Input Motion"
        length = model_kwargs["y"]["lengths"][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = "input_motion{:02d}.mp4".format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption, dataset=args.dataset, fps=fps, vis_mode="gt", frame_colors=frame_colors.get(sample_i, []))
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            if caption == "":
                caption = "Edit [{}] unconditioned".format(args.edit_mode)
            else:
                caption = "Edit [{}]: {}".format(args.edit_mode, caption)
            length = all_lengths[rep_i * args.batch_size + sample_i]

            print(f"length: {length}")

            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = "sample{:02d}_rep{:02d}.mp4".format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption, dataset=args.dataset, fps=fps, vis_mode=args.edit_mode, frame_colors=frame_colors.get(sample_i, []))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, "sample{:02d}.mp4".format(sample_i))
        ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
        hstack_args = f" -filter_complex hstack=inputs={args.num_repetitions+1}"
        ffmpeg_rep_cmd = f"ffmpeg -y -loglevel warning " + "".join(ffmpeg_rep_files) + f"{hstack_args} {all_rep_save_file}"
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


if __name__ == "__main__":
    main()
