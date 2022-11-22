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


def filter_text(text):
    return "run" in text
    # return True


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
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split="test", hml_mode="train", shuffle=True)  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    all_motions = []

    iterator = iter(data)

    counter = 0

    for input_motions, model_kwargs in iterator:
        # input_motions = input_motions.to(dist_util.dev())
        # Recover XYZ *positions* from HumanML3D vector representation
        if args.dataset == "humanml":
            n_joints = 22 if input_motions.shape[1] == 263 else 21
            n_feats = 1
            input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            input_motions = recover_from_ric(input_motions, n_joints)
            input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

        for sample_i in range(args.num_samples):
            text = model_kwargs["y"]["text"][sample_i]
            length = model_kwargs["y"]["lengths"][sample_i]

            if filter_text(text):
                all_motions.append((input_motions[sample_i], text, length))
                counter += 1

        if counter >= 10:
            break

    # all_motions = np.concatenate(all_motions, axis=0)
    # all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    # all_text = all_text[:total_num_samples]
    # all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # npy_path = os.path.join(out_path, "results.npy")
    # print(f"saving results file to [{npy_path}]")
    # np.save(npy_path, {"motion": all_motions, "text": all_text, "lengths": all_lengths, "num_samples": args.num_samples, "num_repetitions": args.num_repetitions})
    # with open(npy_path.replace(".npy", ".txt"), "w") as fw:
    #     fw.write("\n".join(all_text))
    # with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
    #     fw.write("\n".join([str(l) for l in all_lengths]))

    # print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == "kit" else paramUtil.t2m_kinematic_chain

    for motion, text, length in all_motions:
        caption = text

        print(f"text : {text}")

        motion = motion.transpose(2, 0, 1)[:length]
        save_file = f"{text}.mp4"
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption, dataset=args.dataset, fps=fps, vis_mode="gt")

        all_rep_save_file = os.path.join(out_path, f"{text}.mp4")
        ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
        hstack_args = f" -filter_complex hstack=inputs={args.num_repetitions+1}"
        ffmpeg_rep_cmd = f"ffmpeg -y -loglevel warning " + "".join(ffmpeg_rep_files) + f"{hstack_args} {all_rep_save_file}"
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


if __name__ == "__main__":
    main()
