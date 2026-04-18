"""Shared Plotly save/show helper for analysis modules."""
import os


def save_and_show_plotly(fig, save_images: bool, show_images: bool,
                         output_dir, filename: str,
                         width: int = 1080, height: int = 720, scale: int = 2,
                         save_html: bool = False):
    """Save a Plotly figure to PNG (and optionally HTML) and/or display it.

    Args:
        fig: Plotly figure object.
        save_images: Whether to write the figure to disk.
        show_images: Whether to display the figure interactively.
        output_dir: Directory to write files into.
        filename: Base filename without extension (e.g. 'ACN_charges_by_hour').
        width, height, scale: Image export dimensions.
        save_html: If True, also export an HTML version.
    """
    if save_images:
        fig.write_image(os.path.join(str(output_dir), f'{filename}.png'),
                        width=width, height=height, scale=scale)
        if save_html:
            fig.write_html(os.path.join(str(output_dir), f'{filename}.html'))
    if show_images:
        fig.show()
