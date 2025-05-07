import ome_types
from pathlib import Path
import pyvips


current_script_path = Path(__file__)
xml_file_path = str(current_script_path.parent.parent / "default_pred_ome_config.xml")
DEFAUT_CONFIG = ome_types.from_xml(xml_file_path)


PYVIPS2OME_FORMAT = {
    pyvips.BandFormat.UCHAR: "uint8",      # Unsigned 8-bit integer
    pyvips.BandFormat.CHAR: "int8",        # Signed 8-bit integer
    pyvips.BandFormat.USHORT: "uint16",    # Unsigned 16-bit integer
    pyvips.BandFormat.SHORT: "int16",      # Signed 16-bit integer
    pyvips.BandFormat.UINT: "uint32",      # Unsigned 32-bit integer
    pyvips.BandFormat.INT: "int32",        # Signed 32-bit integer
    pyvips.BandFormat.FLOAT: "float",      # 32-bit floating point
    pyvips.BandFormat.DOUBLE: "double",    # 64-bit floating point
    pyvips.BandFormat.COMPLEX: "complex64",# Complex number, 2x 32-bit floating point
    pyvips.BandFormat.DPCOMPLEX: "complex128", # Complex number, 2x 64-bit floating point
}


def adapt_ome_metadata(final_img, resolution, channel_names, magnification):

    xml_config = DEFAUT_CONFIG
    n_channels = len(channel_names)

    ome_format = PYVIPS2OME_FORMAT[final_img.format]
    assert final_img.height % n_channels == 0

    xml_config.images[0].pixels.size_c = n_channels
    xml_config.images[0].pixels.type = ome_format
    xml_config.images[0].pixels.size_x = final_img.width
    xml_config.images[0].pixels.size_y = final_img.height // n_channels
    xml_config.images[0].pixels.physical_size_x = resolution
    xml_config.images[0].pixels.physical_size_y = resolution

    planes = [ome_types.model.Plane(the_z=0, the_t=0, the_c=idx_c) for idx_c in range(n_channels)]
    channels = [ome_types.model.Channel(
        id=f'Channel:{idx_c}', name=channel_names[idx_c], samples_per_pixel=1,
        light_path={}) for idx_c in range(n_channels)]
    xml_config.images[0].pixels.planes = planes
    xml_config.images[0].pixels.channels = channels
    xml_config.instruments[0].objectives[0].nominal_magnification = magnification
    return xml_config.to_xml()
