from pyfibre.model.image_analyser import ImageAnalyser


def process_run(files, p_denoise, sigma, alpha, queue):
    print('process_run')
    print('entering for loop')

    image_analyser = ImageAnalyser(
        p_denoise=p_denoise, sigma=sigma, alpha=alpha)

    for prefix, data in files.items():
        try:
            image_analyser.image_analysis(
                data['image'], prefix)
            print('image_analysis done')
            queue.put("Analysis of {} complete".format(prefix))

        except Exception as err:
            print('something went wrong')
            queue.put("Error occurred in analysis of {}".format(prefix))
            raise err
