"""
Palmer Penguins Inference CLI Tool
Template Repository for MSU AI Club Workshop 01

Usage:
    python predict.py --bill_length 48.5 --bill_depth 15.0 --flipper_length 217 --body_mass 5000
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

# Ensure cross-platform UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Palmer Penguin Species from bill/flipper measurements.")
    parser.add_argument("--bill_length", type=float, default=47.5, help="Bill length in mm (default: 47.5)")
    parser.add_argument("--bill_depth", type=float, default=15.0, help="Bill depth in mm (default: 15.0)")
    parser.add_argument("--flipper_length", type=float, default=217.0, help="Flipper length in mm (default: 217.0)")
    parser.add_argument("--body_mass", type=float, default=5000.0, help="Body mass in grams (default: 5000.0)")
    parser.add_argument("--model_path", type=str, default="penguin_model.pkl", help="Path to saved model artifact")
    return parser.parse_args()

def load_model(model_path):
    """
    Ensure saved model artifact exists and load it cleanly for prediction.
    """
    if not os.path.exists(model_path):
        print(f"[Error] Model artifact '{model_path}' not found.")
        print("        Please run 'python train.py' first to execute the training pipeline!")
        sys.exit(1)
        
    model = None
    # TODO: Load serialized model artifact using pickle.load()
    # YOUR CODE HERE:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def predict_species(model, bill_len, bill_dep, flipper_len, body_mass):
    """
    Construct input DataFrame matching feature schema and compute species prediction.
    """
    if model is None:
        print("⚠️ Model is None! Please complete load_model() with pickle.load() first.")
        return None

    # TODO: Construct input pandas DataFrame matching exact feature column names
    input_df = pd.DataFrame({
        "bill_length_mm": [bill_len],
        "bill_depth_mm": [bill_dep],
        "flipper_length_mm": [flipper_len],
        "body_mass_g": [body_mass]
    })

    if input_df is not None:
        # TODO: Predict species label and class probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        
        if prediction is not None and probabilities is not None:
            print("\n" + "=" * 55)
            print("PALMER PENGUINS SPECIES PREDICTOR")
            print("=" * 55)
            print("Input Measurements:")
            print(f"  * Bill Length    : {bill_len:.1f} mm")
            print(f"  * Bill Depth     : {bill_dep:.1f} mm")
            print(f"  * Flipper Length : {flipper_len:.1f} mm")
            print(f"  * Body Mass      : {body_mass:.0f} g")
            print("-" * 55)
            print(f"PREDICTED SPECIES  : >>> {prediction.upper()} <<<")
            print("-" * 55)
            print("Probability Distribution:")
            for cls, prob in zip(classes, probabilities):
                bar = "#" * int(prob * 20)
                print(f"  * {cls:10s} : {prob:6.1%}  [{bar:<20s}]")
            print("=" * 55 + "\n")
            return prediction
        else:
            print("💡 Complete prediction and probabilities variables above!")
    else:
        print("💡 Complete input_df DataFrame creation above!")

def main():
    args = parse_args()
    model = load_model(args.model_path)
    predict_species(model, args.bill_length, args.bill_depth, args.flipper_length, args.body_mass)

if __name__ == "__main__":
    main()
