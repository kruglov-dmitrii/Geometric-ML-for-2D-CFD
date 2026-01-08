"""Example programmatic runner for TFRecord->.pt conversion.

Import `convert_tfrecord_to_pt` from `scripts.convert_tfrecord_to_pt` and call it
from your code to customize behavior.

Usage: modify the variables below and run `python scripts/run_conversion.py`.
"""
from convert_tfrecord_to_pt import convert_tfrecord_to_pt

# TFRECORD = "./deepmind_datasets/cylinder_flow/cylinder_flow/test.tfrecord"
# META = "./deepmind_datasets/cylinder_flow/cylinder_flow/meta.json"
# OUT_DIR = "deepmind_datasets/cylinder_flow/cylinder_flow/pt_test"

# if __name__ == "__main__":
#     # convert entire file; set max_samples to an int to limit
#     files = convert_tfrecord_to_pt(TFRECORD, META, OUT_DIR, max_samples=None)
#     print(f"Saved {len(files)} samples to {OUT_DIR}")


TF_PATHS = ['./deepmind_datasets/cylinder_flow/cylinder_flow/test.tfrecord', 
            './deepmind_datasets/cylinder_flow/cylinder_flow/valid.tfrecord', 
            './deepmind_datasets/cylinder_flow/cylinder_flow/train.tfrecord'
            ]
META = ["./deepmind_datasets/cylinder_flow/cylinder_flow/meta.json"] * 3
PT_PATHS = ['./deepmind_datasets/cylinder_flow/cylinder_flow/pt_test', 
            './deepmind_datasets/cylinder_flow/cylinder_flow/pt_valid', 
            './deepmind_datasets/cylinder_flow/cylinder_flow/pt_train'
            ]

if __name__ == "__main__":
    for TFRECORD, META, OUT_DIR in zip(TF_PATHS, META, PT_PATHS):
        files = convert_tfrecord_to_pt(TFRECORD, META, OUT_DIR, max_samples=None)
        print(f"Saved {len(files)} samples to {OUT_DIR}")