import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_calibration_data_loader,get_unest_calibration_data_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=20,
        help="Number of images to use in calibration.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to save the image tensor data in FP16 format."
    )
    parser.add_argument(
        "--output_path", type=str, default="calib_btcv.npy", help="Path to output npy file."
    )
    parser.add_argument(
        "--data_path", type=str, default=".", help="Path to the directory containing images."
    )
    parser.add_argument(
        "--label_path", type=str, default=".", help="Path to the directory containing images."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for DataLoader."
    )
    parser.add_argument('--roi_width', type=int, default=96,
                        help='ROI width for the input.')
    parser.add_argument('--roi_height', type=int, default=96,
                        help='ROI height for the input.')
    parser.add_argument('--roi_depth', type=int, default=96,
                        help='ROI depth for the input.')
    parser.add_argument('--net', type=str, default='unet',
                        choices=['unet', 'vit', 'unetr', 'swinunetr','unest'],
                        help='Net to export')
    parser.add_argument('--a_min', type=float, default=-175.0,
                        help='Minimum intensity value for scaling.')
    parser.add_argument('--a_max', type=float, default=250.0,
                        help='Maximum intensity value for scaling.')
    parser.add_argument('--b_min', type=float, default=0.0,
                        help='Minimum output intensity value after scaling.')
    parser.add_argument('--b_max', type=float, default=1.0,
                        help='Maximum output intensity value after scaling.')
    args = parser.parse_args()

    # Use the get_calibration_data_loader from utils.py
    if args.net=='unest':
        data_loader = get_unest_calibration_data_loader(
            data_path=args.data_path,
            label_path=None,
            batch_size=1,
            shuffle=False,  # Set to False to not shuffle
            num_workers=4,
        )
    else:
        data_loader = get_calibration_data_loader(
                data_path=args.data_path,
                label_path= args.label_path,
                batch_size=1,
                shuffle=False,  # Set to False to not shuffle
                num_workers=4,
                roi_depth=args.roi_depth,
                roi_height=args.roi_height,
                roi_width=args.roi_width,
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                num_samples_per_image=1,
            )


    calib_tensor = []
    total_images = 0

    for batch_data in data_loader:
        image = batch_data['image']
        if args.fp16:
            image = image.half()
        calib_tensor.append(image.numpy())
        total_images += image.shape[0]
        if total_images >= args.calibration_data_size:
            break

    # Concatenate and truncate to the desired number of images
    calib_tensor = np.concatenate(calib_tensor, axis=0)[:args.calibration_data_size]
    print(calib_tensor.shape)

    np.save(args.output_path, calib_tensor)
    print(f"Calibration data saved to {args.output_path}")
    print(f"Total images saved: {calib_tensor.shape[0]}")

if __name__ == "__main__":
    main()
