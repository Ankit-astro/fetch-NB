#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from fetch.data_sequence import DataGenerator
from fetch.utils import get_model

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-n", "--nproc", help="Number of processors for training", default=4, type=int
    )
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action='append'
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="Directory to save the results.",
        required=False,
        type=str
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for training data", default=8, type=int
    )
    parser.add_argument(
        "-m", "--model", help="Path to the model file (.keras or original)", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.nproc > 1:
        use_multiprocessing = True
        logging.info(f"Using multiprocessing with {args.nproc} workers")
    else:
        use_multiprocessing = False

    if args.model.endswith(".keras"):
        model = tf.keras.models.load_model(args.model)
    else:
        model = get_model(args.model)

    for data_dir in args.data_dir:

        cands_to_eval = glob.glob(f"{data_dir}/*h5")

        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue

        logging.debug(f"Read {len(cands_to_eval)} candidates")

        # Get the data generator, make sure noise and shuffle are off.
        cand_datagen = DataGenerator(
            list_IDs=cands_to_eval,
            labels=[0] * len(cands_to_eval),
            shuffle=False,
            noise=False,
            batch_size=args.batch_size,
        )

        # get's get predicting
        probs = model.predict(
            x=cand_datagen,
            verbose=1,
            use_multiprocessing=use_multiprocessing,
            workers=args.nproc,
            steps=len(cand_datagen),
        )

        # Save results
        output_dir = args.out_dir if args.out_dir else data_dir
        results_dict = {}
        results_dict["candidate"] = cands_to_eval
        results_dict["probability"] = probs[:, 1]
        results_dict["label"] = np.round(probs[:, 1] >= args.probability)
        results_file = os.path.join(output_dir,f"results_{os.path.basename(args.model).replace('.keras', '').replace('.h5', '')}.csv")
        pd.DataFrame(results_dict).to_csv(results_file)
