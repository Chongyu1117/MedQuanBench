import argparse
import torch
import numpy as np
import warnings
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    Compose,
)
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# Import utility functions
from utils import get_data_loader, dice  # Make sure these are available

# Import model optimization and deployment utilities
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes


def benchmark(
    args,
    predictor,
    data_loader,
    input_shape,
    compute_dice=False,
    num_classes=2,
):
    """
    Perform inference with the deployed model using MONAI's sliding window.
    Optionally compute Dice score, if 'compute_dice' is True.
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dice_list_case = []

    for i, batch in enumerate(data_loader):
        input_data = batch["image"].to(device)

        outputs_data = sliding_window_inference(
            inputs=input_data,
            roi_size=input_shape,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.5,
            mode="gaussian",
            device="cpu",
        )

        if compute_dice:
            labels = batch["label"].to(device)
            outputs_seg = torch.softmax(outputs_data, dim=1).cpu().numpy()
            outputs_seg = np.argmax(outputs_seg, axis=1).astype(np.uint8)
            labels = labels.cpu().numpy()[:, 0, :, :, :]

            dice_list_sub = []
            for organ_idx in range(1, num_classes):
                organ_dice_val = dice(outputs_seg[0] == organ_idx, labels[0] == organ_idx)
                dice_list_sub.append(organ_dice_val)

            mean_dice = np.mean(dice_list_sub)
            print(f"Mean Organ Dice for sample {i+1}: {mean_dice:.4f}")
            dice_list_case.append(mean_dice)

    if compute_dice and len(dice_list_case) > 0:
        overall_mean_dice = np.mean(dice_list_case)
        print(f"\nOverall Mean Dice across samples: {overall_mean_dice:.4f}")

    print("Inference completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deployed UNet/TransUNet model (no saving, no time measurement)"
    )
    parser.add_argument("--onnx_path", type=str, default=".", help="Path to the ONNX model.")
    parser.add_argument("--data_path", type=str, default=".", help="Path to the images.")
    parser.add_argument("--label_path", type=str, default=".", help="Path to the labels.")
    parser.add_argument("--roi_width", type=int, default=96, help="ROI width for the input.")
    parser.add_argument("--roi_height", type=int, default=96, help="ROI height for the input.")
    parser.add_argument("--roi_depth", type=int, default=96, help="ROI depth for the input.")
    parser.add_argument("--compute_dice", action="store_true", help="Compute Dice score.")
    parser.add_argument(
        "--net",
        type=str,
        default="transunet",
        choices=["unet", "transunet"],
        help="Only unet or transunet are supported here",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="stronglyTyped",
        choices=["fp16", "fp32", "int8", "bf16", "best", "stronglyTyped"],
        help="Quantization mode for the TensorRT engine.",
    )
    parser.add_argument("--num_classes", type=int, default=14, help="Number of Classes.")
    parser.add_argument("--a_min", type=float, default=-175.0, help="Minimum intensity value for scaling.")
    parser.add_argument("--a_max", type=float, default=250.0, help="Maximum intensity value for scaling.")
    parser.add_argument("--b_min", type=float, default=0.0, help="Minimum output intensity value after scaling.")
    parser.add_argument("--b_max", type=float, default=1.0, help="Maximum output intensity value after scaling.")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

    # Configure deployment
    deployment = {
        "runtime": "TRT",
        "version": "10.3",
        "precision": args.quantize_mode,
    }

    # Create an ONNX bytes object
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # Get the runtime client
    client = RuntimeRegistry.get(deployment)

    # Compile the TRT model
    print("Compiling the TensorRT engine. This may take a few minutes...")
    compiled_model = client.ir_to_compiled(onnx_bytes)
    print("Compilation completed.")

    # Print size of the compiled model
    engine_size = len(compiled_model)
    print(f"Size of the TensorRT engine: {engine_size / (1024 ** 2):.2f} MB")

    # Create the device model
    device_model = DeviceModel(client, compiled_model, metadata={})
    print(f"Inference latency reported by device_model: {device_model.get_latency()} ms")

    keys = ["image", "label"] if args.compute_dice else ["image"]
    preprocessing_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=[1.5, 1.5, 2.0],
            mode=("bilinear", "nearest") if args.compute_dice else "bilinear",
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        EnsureTyped(keys=keys, track_meta=True, data_type="tensor"),
    ])

    # Prepare DataLoader
    data_loader = get_data_loader(
        data_path=args.data_path,
        label_path=args.label_path if args.compute_dice else None,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        roi_depth=args.roi_depth,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        num_samples_per_image=1,
        transforms=preprocessing_transforms,
    )

    def predictor(input_tensor):
        input_tensor_cpu = input_tensor.cpu()
        outputs = device_model(input_tensor_cpu)
        if isinstance(outputs, torch.Tensor):
            return outputs.to(input_tensor.device)
        elif isinstance(outputs, (list, tuple)):
            return outputs[0].to(input_tensor.device)
        else:
            raise TypeError(f"Unexpected output type from device_model: {type(outputs)}")

    benchmark(
        args,
        predictor=predictor,
        data_loader=data_loader,
        input_shape=(args.roi_width, args.roi_height, args.roi_depth),
        compute_dice=args.compute_dice,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    set_determinism(seed=42)
    main()
