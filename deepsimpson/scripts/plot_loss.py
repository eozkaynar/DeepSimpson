import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

models = {
    "LSTM": {"epochs": 100, "skip_first_loss": True},
    "RNN": {"epochs": 70, "skip_first_loss": False}
}

for model_name, settings in models.items():
    print(f"Processing model: {model_name}")
    epochs = settings["epochs"]
    skip_first = settings["skip_first_loss"]

    # Load losses
    train_losses = np.load(f"output/prediction/_{model_name}/train_losses.npy")
    val_losses   = np.load(f"output/prediction/_{model_name}/val_losses.npy")

    if skip_first:
        train_losses = train_losses[1:]
        val_losses = val_losses[1:]

    iterations = np.linspace(0, epochs, num=len(train_losses))
    iterations_val = np.linspace(0, epochs, num=len(val_losses))

    # Load predictions
    df = pd.read_csv(f"output/prediction/_{model_name}/test_predictions.csv")

    # EF scatter plot with risk zones
    plt.figure(figsize=(10, 6))
    true_vals = df['true_value'] * 100
    pred_vals = df['prediction'] * 100

    # Background risk zones
    plt.axvspan(0, 30, color='red', alpha=0.1, label='Severely Reduced EF (<30%)')
    plt.axvspan(30, 50, color='orange', alpha=0.1, label='Moderately Reduced EF (30–50%)')
    plt.axvspan(50, 70, color='green', alpha=0.1, label='Normal EF (50–70%)')
    plt.axvspan(70, 100, color='gray', alpha=0.05, label='Elevated EF (>70%)')
    plt.axhspan(0, 30, color='red', alpha=0.1)
    plt.axhspan(30, 50, color='orange', alpha=0.1)
    plt.axhspan(50, 70, color='green', alpha=0.1)
    plt.axhspan(70, 100, color='gray', alpha=0.05)

    # Scatter points and perfect fit line
    plt.scatter(true_vals, pred_vals, color='blue', label='True vs Prediction', alpha=0.6)
    x_vals = np.linspace(0, 100, 100)
    plt.plot(x_vals, x_vals, color='black', linestyle='--', label='Perfect Fit (y = x)')

    # Labels
    plt.xlabel('True EF (%)')
    plt.ylabel('Predicted EF (%)')
    plt.title(f'EF Prediction vs Ground Truth ({model_name})')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"output_images/EF_plot_with_risk_zones_{model_name}.png")
    plt.show()

    # Training/Validation Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label="Training Loss")
    plt.plot(iterations_val, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss Over Epochs ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output_images/training_validation_loss_plot_{model_name}.png")
    plt.show()
