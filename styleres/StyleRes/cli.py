#!/usr/bin/env python3
import argparse
from PIL import Image
import os
from inference import initialize_styleres
from utils import AppUtils
from datasets.process_image import ImageProcessor

def main():
    parser = argparse.ArgumentParser(description='StyleRes CLI - Transform images using StyleGAN residuals')
    parser.add_argument('--input-image', type=str, help='Path to input image')
    parser.add_argument('--output-image', type=str, help='Path to save output image')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to use (cpu/cuda)')
    parser.add_argument('--method', type=str, default=None, help='Method to use for editing')
    parser.add_argument('--edit', type=str, default=None, help='Type of edit to apply')
    parser.add_argument('--factor', type=float, default=0.0, help='Strength of the edit')
    parser.add_argument('--align', action='store_true', help='Crop and align face before processing')
    parser.add_argument('--list-methods', action='store_true', help='List available methods and exit')
    parser.add_argument('--list-edits', type=str, help='List available edits for a specific method and exit')
    
    args = parser.parse_args()
    
    # Initialize utilities
    utils = AppUtils()
    methods = utils.get_methods()
    
    # Handle listing options
    if args.list_methods:
        print("Available methods:")
        for method in methods:
            print(f"  - {method}")
        return
    
    if args.list_edits:
        if args.list_edits not in methods:
            print(f"Error: Method '{args.list_edits}' not found.")
            print("Available methods:", ", ".join(methods))
            return
        edits = utils.get_edits(args.list_edits)
        print(f"Available edits for method '{args.list_edits}':")
        for edit in edits:
            print(f"  - {edit}")
        return
    
    # Validate required arguments
    if not args.input_image or not args.output_image:
        parser.error("Input and output image paths are required")
    
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return
    
    # Set defaults if not provided
    method = args.method or methods[0]
    if method not in methods:
        print(f"Error: Method '{method}' not found.")
        print("Available methods:", ", ".join(methods))
        return
    
    edits = utils.get_edits(method)
    edit = args.edit or edits[0]
    if edit not in edits:
        print(f"Error: Edit '{edit}' not available for method '{method}'.")
        print("Available edits:", ", ".join(edits))
        return
    
    # Validate factor range
    try:
        minimum, maximum, step = utils.get_range(method)
        if minimum is not None and maximum is not None:
            if not (minimum <= args.factor <= maximum):
                print(f"Error: Factor {args.factor} is out of range [{minimum}, {maximum}] for method '{method}'")
                return
    except (TypeError, ValueError):
        print(f"Warning: Could not get factor range for method '{method}', proceeding with factor {args.factor}")
    
    print(f"Processing image with:")
    print(f"  Method: {method}")
    print(f"  Edit: {edit}")
    print(f"  Factor: {args.factor}")
    print(f"  Align: {args.align}")
    print(f"  Device: {args.device}")
    
    # Initialize model and processor
    print("Loading model...")
    styleres = initialize_styleres('checkpoints/styleres_ffhq.pth', args.device)
    image_processor = ImageProcessor('checkpoints/shape_predictor_68_face_landmarks.dat')
    
    # Load and process image
    print("Loading input image...")
    try:
        image = Image.open(args.input_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print("Processing image...")
    try:
        # Create configuration
        cfg = utils.args_to_cfg(method, edit, args.factor)
        
        # Align face if requested
        if args.align:
            print("Aligning face...")
            image = image_processor.align_face(image)
        
        # Preprocess image
        image = image_processor.preprocess_image(image, is_batch=False)
        
        # Apply edit
        image = styleres.edit_images(image, cfg)
        
        # Postprocess image
        image = image_processor.postprocess_image(image.detach().cpu().numpy(), is_batch=False)
        
        # Save result
        print(f"Saving result to {args.output_image}...")
        image.save(args.output_image)
        print("Done!")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

if __name__ == "__main__":
    main()