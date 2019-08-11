from pyfibre.model.image_analysis import image_analysis


def process_run(files, p_denoise, sigma, alpha, queue):
    print('process_run')
    print('entering for loop')
    for prefix, data in files.items():
        try:
            image_analysis(
                data['image'], prefix, scale=1.25,
                sigma=sigma, alpha=alpha, p_denoise=p_denoise
            )
            print('image_analysis done')
            queue.put("Analysis of {} complete".format(prefix))

        except Exception as err:
            print('something went wrong')
            queue.put("Error occurred in analysis of {}".format(prefix))
            raise err
