import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def show_inference_window(
    image_uint8: np.ndarray,
    predicted_class: str,
    top5: list[tuple[str, float]],
    correct_class: str | None,
    window_title: str = "Inference",
) -> None:
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("Expected image_uint8 as HxWx3 RGB array")

    fig = plt.figure(figsize=(12, 6))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[3.0, 2.0],
        height_ratios=[5.0, 1.2],
        wspace=0.2,
        hspace=0.15,
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_uint8)
    ax_img.set_axis_off()

    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.set_axis_off()
    ax_text.text(
        0.5,
        0.5,
        f"I think this is a {predicted_class}!!!.",
        ha="center",
        va="center",
        fontsize=14,
    )

    ax_scores = fig.add_subplot(gs[:, 1])
    ax_scores.set_axis_off()
    ax_scores.text(0.0, 1.0, "Top 5 scores", ha="left", va="top", fontsize=13)

    y = 0.88
    for class_name, score in top5:
        is_correct = correct_class is not None and class_name == correct_class
        ax_scores.text(
            0.0,
            y,
            f"{class_name}",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold" if is_correct else "normal",
        )
        ax_scores.text(
            1.0,
            y,
            f"{score:.4f}",
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold" if is_correct else "normal",
        )
        y -= 0.11

    plt.show(block=True)
