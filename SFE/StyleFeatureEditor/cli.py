import os
import sys
import argparse

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

from runners.simple_runner import SimpleRunner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("✅ All imports successful! Ready to use StyleFeatureEditor.")

def plot_one_image(fig, pth, title, subplot_args):
  img = Image.open(pth)
  ax = fig.add_subplot(*subplot_args)
  ax.imshow(img)
  ax.set_title(title, fontsize=20)
  ax.axis('off')

def plot_edited_images(orig_pth, edited_pth, inversion_pth, e4e_inv_pth=None, e4e_edit_pth=None, unaligned_path=None):
    if unaligned_path is not None:
      fig_size_y = 2 + (e4e_inv_pth is not None or e4e_edit_pth is not None)
      fig_size_x = 2
    else:
      fig_size_y = 1 + (e4e_inv_pth is not None or e4e_edit_pth is not None)
      fig_size_x = 2 + (inversion_pth is not None)
    img_num = 1

    grid = gridspec.GridSpec(fig_size_y, fig_size_x)
    fig = plt.figure(figsize=(16, 8))

    plot_one_image(fig, orig_pth, "Original Image", (fig_size_y, fig_size_x, img_num))
    img_num += 1

    if unaligned_path is not None:
      plot_one_image(fig, unaligned_path, "Unaligned Image", (fig_size_y, fig_size_x, img_num))
      img_num += 1

    plot_one_image(fig, inversion_pth, "Reconstructed Image", (fig_size_y, fig_size_x, img_num))
    img_num += 1

    plot_one_image(fig, edited_pth, "Edited Image", (fig_size_y, fig_size_x, img_num))
    img_num += 1

    if e4e_inv_pth is not None:
      plot_one_image(fig, e4e_inv_pth, "e4e inversion", (fig_size_y, fig_size_x, img_num))
      img_num += 1

    if e4e_inv_pth is not None:
      plot_one_image(fig, e4e_edit_pth, "e4e editing", (fig_size_y, fig_size_x, img_num))
      img_num += 1

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='StyleFeatureEditor test script')
    
    
    parser.add_argument('--input', '-i', type=str, default='assets/dicaprio.png',
                        help='Path to input image (default: assets/dicaprio.png)')
    parser.add_argument('--output', '-o', type=str, default='editing_res/dicaprio.png',
                        help='Path to save edited image (default: editing_res/dicaprio.png)')
    
    
    parser.add_argument('--editing', '-e', type=str, default='age',
                        help='Type of editing to apply (default: age)')
    parser.add_argument('--power', '-p', type=float, default=9.0,
                        help='Editing power/strength (default: 9.0)')
    
    
    parser.add_argument('--model', '-m', type=str, default='pretrained_models/sfe_editor_light.pt',
                        help='Path to model checkpoint (default: pretrained_models/sfe_editor_light.pt)')
    
    
    parser.add_argument('--no-align', action='store_true',
                        help='Disable image alignment (default: align enabled)')
    parser.add_argument('--no-save-inversion', action='store_true',
                        help='Disable saving inversion (default: save inversion enabled)')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable mask usage (default: mask enabled)')
    parser.add_argument('--mask-threshold', type=float, default=0.095,
                        help='Mask threshold value (default: 0.095)')
    
    
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting results (default: plot enabled)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    runner = SimpleRunner(
        editor_ckpt_pth=args.model
    )
    
    
    runner.edit(
        orig_img_pth=args.input,
        editing_name=args.editing,
        edited_power=args.power,
        save_pth=args.output,
        align=not args.no_align,
        save_inversion=not args.no_save_inversion,
        use_mask=not args.no_mask,
        mask_trashold=args.mask_threshold
    )
    
    
    if not args.no_plot:
        
        base_name = os.path.splitext(args.output)[0]
        inversion_path = f"{base_name}_inversion.jpg"
        unaligned_path = f"{base_name}_unaligned.jpg" if not args.no_align else None
        
        plot_edited_images(
            orig_pth=args.input,
            edited_pth=args.output,
            inversion_pth=inversion_path,
            unaligned_path=unaligned_path
        )

if __name__ == "__main__":
    main()