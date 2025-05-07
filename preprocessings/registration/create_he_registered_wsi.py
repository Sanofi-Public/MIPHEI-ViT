from pathlib import Path
import pandas as pd
from tqdm import tqdm
from valis import registration
import gc

DATAFRAME_PATH = "/root/workdir/Immucan/dataframe.csv"
REGISTRATION_DIR = "/root/workdir/valis/out_immucan"
OUT_DIR = "/root/workdir/valis/he_registered"

if __name__ == "__main__":
    Path(OUT_DIR).mkdir(exist_ok=True)
    dataframe = pd.read_csv(DATAFRAME_PATH)
    pickle_paths = [str(fn) for fn in Path(REGISTRATION_DIR).rglob("*.pickle")]
    for pickle_path in tqdm(pickle_paths):
        registrar = registration.load_registrar(pickle_path)
        for slide_name_key in registrar.slide_dict.keys():
            if "HES" in slide_name_key:
                break

        slide_obj = registrar.slide_dict[slide_name_key]
        dst_f = str(Path(OUT_DIR) / (slide_name_key + ".ome.tiff"))
        if Path(dst_f).exists():
            continue

        slide_cmap = None
        is_rgb = slide_obj.reader.metadata.is_rgb
        if is_rgb:
            updated_channel_names = None
        else:
            raise ValueError



        slide_obj.warp_and_save_slide(
            dst_f=dst_f, level=0,
            non_rigid=True,
            crop=True,
            src_f=slide_obj.src_f,
            interp_method="bicubic",
            colormap=slide_cmap,
            tile_wh=512,
            compression="deflate",
            channel_names=updated_channel_names,
            Q=100,
            pyramid=True)
        del slide_obj, registrar
        registration.kill_jvm()
        gc.collect()
