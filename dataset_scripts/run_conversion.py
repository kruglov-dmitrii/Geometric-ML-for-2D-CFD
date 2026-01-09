from convert_tfrecord_to_pt import convert_tfrecord_to_pt

# Location of the DeepMind TFRecord files and metadata
TF_PATHS = ['./deepmind_datasets/cylinder_flow/test.tfrecord', 
            './deepmind_datasets/cylinder_flow/valid.tfrecord', 
            './deepmind_datasets/cylinder_flow/train.tfrecord'
            ]
# Location of the DeepMind metadata file
META = ["./deepmind_datasets/cylinder_flow/meta.json"] * 3
# Output location for the converted PyTorch files
PT_PATHS = ['./cylinder_flow_dataset/test', 
            './cylinder_flow_dataset/valid', 
            './cylinder_flow_dataset/train'
            ]

if __name__ == "__main__":
    for TFRECORD, META, OUT_DIR in zip(TF_PATHS, META, PT_PATHS):
        files = convert_tfrecord_to_pt(TFRECORD, META, OUT_DIR, max_samples=None)
        print(f"Saved {len(files)} samples to {OUT_DIR}")