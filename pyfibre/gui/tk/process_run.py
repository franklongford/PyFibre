from pyfibre.io.tif_reader import TIFReader
from pyfibre.model.image_analysis import image_analysis


def run_analysis(input_files, p_intensity, p_denoise,
                 sigma, alpha, ow_metric, ow_segment, ow_network,
                 ow_figure, queue):

    reader = TIFReader(input_files,
                       shg=True, pl=True,
                       p_intensity=p_intensity,
                       ow_network=ow_network, ow_segment=ow_segment,
                       ow_metric=ow_metric, ow_figure=ow_figure)
    reader.load_multi_images()

    for prefix, data in reader.files.items():
        try:
            image_analysis(
                data['image'], prefix, scale=1.25,
                sigma=sigma, alpha=alpha, p_denoise=p_denoise
            )
            queue.put("Analysis of {} complete".format(prefix))

        except Exception as err:
            queue.put("Error occurred in analysis of {}".format(prefix))
            raise err
