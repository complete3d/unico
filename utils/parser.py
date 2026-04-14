def get_args():
    raise RuntimeError(
        "The argparse entrypoints have been removed. "
        "Use the helper scripts with Hydra overrides instead, for example: "
        "`CUDA_VISIBLE_DEVICES=0 ./scripts/train_dp.sh experiment=abcmulti`."
    )
