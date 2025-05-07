from valis import registration
import argparse

import traceback
from valis import slide_io, valtils
from valis.registration import DEFAULT_COMPRESSION
from pathlib import Path
import os

def valis_apply_nuclei(registrar, src_f, nuclei_src_f, dst_dir,
                       non_rigid=True, crop=True, tile_wh=None):

    cmap_is_str = False
    named_color_map = None
    colormap = slide_io.CMAP_AUTO
    if isinstance(colormap, str) and colormap == slide_io.CMAP_AUTO:
        cmap_is_str = True
    else:
        named_color_map = {registrar.get_slide(x).name:colormap[x] for x in colormap.keys()}

    slide_obj = registrar.get_slide(src_f)
    reader_cls = slide_io.get_slide_reader(nuclei_src_f, series=None) #Get appropriate slide reader class
    slide_nuclei = reader_cls(nuclei_src_f, series=None)
    updated_channel_names = slide_nuclei.metadata.channel_names
    slide_cmap = None
    try:
        if not cmap_is_str and named_color_map is not None:
            slide_cmap = named_color_map[slide_obj.name]
        else:
            slide_cmap = colormap

        slide_cmap = slide_io.check_colormap(colormap=slide_cmap, channel_names=updated_channel_names)
    except Exception as e:
        traceback_msg = traceback.format_exc()
        msg = f"Could not create colormap for the following reason:{e}"
        valtils.print_warning(msg, traceback_msg=traceback_msg)

    dst_f = os.path.join(dst_dir, Path(nuclei_src_f).stem + ".tiff")

    slide_obj.warp_and_save_slide(dst_f=dst_f, level=0,
                                    non_rigid=non_rigid,
                                    crop=crop,
                                    src_f=nuclei_src_f,
                                    interp_method="nearest",
                                    colormap=slide_cmap,
                                    tile_wh=tile_wh,
                                    compression=DEFAULT_COMPRESSION,
                                    channel_names=updated_channel_names,
                                    Q=100,
                                    pyramid=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='src_dir')
    parser.add_argument('--dst_dir', type=str, help='-dst_dir')
    #parser.add_argument('-registered_slide_dst_dir', type=str, help='registered_slide_dst_dir')
    parser.add_argument('--reference_slide', type=str, help='reference_slide')
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    #registered_slide_dst_dir = args.registered_slide_dst_dir
    reference_slide = args.reference_slide

    # Create a Valis object and use it to register the slides in src_dir, aligning *towards* the reference slide.
    registrar = registration.Valis(src_dir, dst_dir, reference_img_f=reference_slide, align_to_reference=False)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Perform micro-registration on higher resolution images, aligning *directly to* the reference image
    #registrar.register_micro(max_non_rigid_registration_dim_px=2000, align_to_reference=False)
    #registrar.warp_and_save_slides(dst_dir, crop="overlap")
    #valis_apply_nuclei(registrar, src_f, nuclei_src_f, dst_dir)
    #registrar.warp_and_save_slides(dst_dir, level=2, crop="overlap")
    

    # Kill the JVM
    registration.kill_jvm()
