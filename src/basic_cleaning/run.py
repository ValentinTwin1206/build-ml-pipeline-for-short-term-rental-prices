#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading artifact from 'Weights&Biases'")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    # Load the data
    logger.info("Parsing downloaded artifact")
    df = pd.read_csv(artifact_local_path)
    
    # Drop outliers using min_price and max_price
    logger.info(f"Filtering prices between 'min={args.min_price}' and 'max={args.max_price}'")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    logger.info(f"Fill missing values with either '0' or empty string")
    df['name'].fillna(value='', inplace=True)
    df['host_name'].fillna(value='', inplace=True)
    df['reviews_per_month'].fillna(value=0, inplace=True)
    df.info()

    # Save the cleaned data
    logger.info(f"Saving cleaned data to '{args.output_artifact}'")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv(args.output_artifact, index=False)
    
    # Assert that the CSV file exists before uploading
    if not os.path.exists(args.output_artifact):
        raise AssertionError(f"Output file '{args.output_artifact}' does not exist")
    
    # Upload the cleaned data as a new artifact
    logger.info("Uploading cleaned data to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

#
# CLI
# # # # #
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact containing raw data to be cleaned",
        required=True
    )
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact after cleaning",
        required=True
    )
    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., clean_data)",
        required=True
    )
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to filter the data (in dollars)",
        required=True
    )
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to filter the data (in dollars)",
        required=True
    )

    args = parser.parse_args()

    go(args)
