import tifffile
import argparse
import zarr
import dask.array as da
import napari
from magicgui import magicgui
from pathlib import Path
import json
import os

# Default indices
channel_idx = 0  # Default main channel index

if __name__ == "__main__":
    # Path to your multi-resolution image
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_path', type=str, help='Slide Path')
    args = parser.parse_args()
    wsi_path = args.wsi_path

    # Load existing channel_settings if the file exists (convert keys to integers, except 'af_idx')
    def load_channel_settings():
        # Ensure the default save path is well-formed
        default_save_path = str(Path("settings") / (Path(wsi_path).stem + "_settings.json"))
        if os.path.exists(default_save_path):
            with open(default_save_path, "r") as file:
                settings = json.load(file)
                print(f"Settings loaded from {default_save_path}: {settings}")
                return settings
        print("No saved settings found. Starting with defaults.")
        return {}  # Return an empty dictionary if the file doesn't exist

    # Process loaded settings to separate af_idx and channel settings
    loaded_settings = load_channel_settings()
    current_idx_af = loaded_settings.pop('af_idx', 0)  # Extract 'af_idx' (default to 0 if not present)
    channel_settings = {int(k): v for k, v in loaded_settings.items()}  # Convert other keys to integers

    # Load the image as a Zarr store
    store = tifffile.imread(wsi_path, aszarr=True)
    cache = zarr.LRUStoreCache(store, max_size=2**30)
    zobj = zarr.open(cache, mode='r')
    data = [
        da.from_zarr(z) for z in [
            zobj[int(dataset['path'])] for dataset in zobj.attrs['multiscales'][0]['datasets']
        ]
    ]

    # Function to get channel data for the given indices
    def get_channels(channel_idx, af_idx):
        channel_data = [d[channel_idx] for d in data]
        af_data = [d[af_idx] for d in data]
        return channel_data, af_data

    # Get initial channels
    channel_data, af_data = get_channels(channel_idx, current_idx_af)

    # Create a Napari viewer
    viewer = napari.Viewer()
    channel_layer = viewer.add_image(
        channel_data, name=f"Channel_{channel_idx}", rgb=False,
        contrast_limits=[0, 65535], colormap="blue", blending="additive",
        visible=False
    )
    af_layer = viewer.add_image(
        af_data, name=f"AutoFluorescence_{current_idx_af}", rgb=False,
        contrast_limits=[0, 65535], colormap="green", blending="additive"
    )
    corrected_data = [
        da.maximum(ch - 0.5 * af, 0).astype("uint16") for ch, af in zip(channel_data, af_data)
    ]
    residual_data = [
        da.maximum(-(ch - 0.5 * af), 0).astype("uint16") for ch, af in zip(channel_data, af_data)
    ]
    residual_layer = viewer.add_image(
        residual_data, name="Residual", rgb=False,
        contrast_limits=[0, 65535], colormap="yellow", blending="additive",
        visible=False
    )
    corrected_layer = viewer.add_image(
        corrected_data, name="Corrected", rgb=False,
        contrast_limits=[0, 65535], colormap="magenta", blending="additive",
        gamma=0.2
    )

    # Function to update the corrected image based on lambda and bias
    def update_correction(lambda_value, bias_value):
        # Save lambda, bias, and af_idx for the current channel in the dictionary (use string keys)
        channel_settings[channel_idx] = {"lambda": lambda_value, "bias": bias_value}
        channel_settings['af_idx'] = current_idx_af
        corrected_data_updated = [
            da.maximum(ch - lambda_value * af + bias_value, 0).astype("uint16")
            for ch, af in zip(channel_data, af_data)
        ]
        residual_data_updated = [
            da.maximum(-(ch - lambda_value * af + bias_value), 0).astype("uint16")
            for ch, af in zip(channel_data, af_data)
        ]
        corrected_layer.data = corrected_data_updated
        residual_layer.data = residual_data_updated

    # Function to update channel layers when channel_idx changes
    def update_channel(new_channel_idx):
        global channel_idx, channel_data, af_data
        channel_idx = new_channel_idx  # Update the global channel index
        channel_data, af_data = get_channels(channel_idx, current_idx_af)
        channel_layer.data = channel_data
        channel_layer.name = f"Channel_{channel_idx}"
        # Check if the current channel has saved settings
        if channel_idx in channel_settings:
            saved_lambda = channel_settings[channel_idx]["lambda"]
            saved_bias = channel_settings[channel_idx]["bias"]
            print(f"Loaded settings for channel {channel_idx}: lambda={saved_lambda}, bias={saved_bias}")
        else:
            saved_lambda = 0.0
            saved_bias = 0.0
            print(f"No settings found for channel {channel_idx}. Using defaults.")

        # Update sliders with saved or default values
        adjust_parameters.lambda_slider.value = saved_lambda
        adjust_parameters.bias_slider.value = saved_bias

        # Force synchronization of sliders
        adjust_parameters.lambda_slider.native.value = saved_lambda
        adjust_parameters.bias_slider.native.value = saved_bias
        update_correction(saved_lambda, saved_bias)

    # Function to update autofluorescence layer when idx_af changes
    def update_af(new_idx_af):
        global current_idx_af, af_data
        current_idx_af = new_idx_af  # Update the dynamic autofluorescence index
        _, af_data = get_channels(channel_idx, current_idx_af)
        af_layer.data = af_data
        af_layer.name = f"AutoFluorescence_{current_idx_af}"
        # Use current slider values to update correction
        update_correction(
            adjust_parameters.lambda_slider.value,
            adjust_parameters.bias_slider.value,
        )

    # Function to save channel_settings as JSON
    def save_channel_settings():
        # Include 'af_idx' in the settings dictionary before saving
        settings_to_save = {str(k): v for k, v in channel_settings.items()}
        settings_to_save['af_idx'] = current_idx_af  # Add 'af_idx'
        default_save_path = str(Path("settings") / (Path(wsi_path).stem + "_settings.json"))

        with open(default_save_path, "w") as file:
            json.dump(settings_to_save, file, indent=4)
        print(f"Settings saved to {default_save_path}")

    # Add sliders for lambda and bias
    @magicgui(
        lambda_slider={"widget_type": "FloatSlider", "min": 0.0, "max": 4.0, "step": 0.01},
        bias_slider={"widget_type": "FloatSlider", "min": -1000.0, "max": 0., "step": 1.0}
    )
    def adjust_parameters(lambda_slider: float = 0.0, bias_slider: float = 0.0):
        update_correction(lambda_slider, bias_slider)

    # Add a widget to select the channel index
    @magicgui(
        channel_idx={"widget_type": "SpinBox", "min": 0, "max": data[0].shape[0] - 1, "step": 1}
    )
    def select_channel(channel_idx: int = 0):
        update_channel(channel_idx)

    # Add a widget to select the autofluorescence index
    @magicgui(
        idx_af={"widget_type": "SpinBox", "min": 0, "max": data[0].shape[0] - 1, "step": 1}
    )
    def select_af(idx_af: int = 0):
        update_af(idx_af)

    # Add a button to save settings
    @magicgui(call_button="Save Settings")
    def save_settings():
        save_channel_settings()

    # Add widgets to Napari viewer
    viewer.window.add_dock_widget(adjust_parameters, name="Parameter Adjustment")
    viewer.window.add_dock_widget(select_channel, name="Channel Selector")
    viewer.window.add_dock_widget(select_af, name="Autofluorescence Selector")
    viewer.window.add_dock_widget(save_settings, name="Save Settings")

    # Connect sliders to update corrected image
    adjust_parameters.lambda_slider.changed.connect(lambda val: update_correction(val, adjust_parameters.bias_slider.value))
    adjust_parameters.bias_slider.changed.connect(lambda val: update_correction(adjust_parameters.lambda_slider.value, val))
    select_af.idx_af.value = current_idx_af

    # Ensure sliders reflect loaded settings for the initial channel
    update_channel(channel_idx)

    napari.run()
    store.close()
