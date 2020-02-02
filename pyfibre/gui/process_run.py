from pyfibre.io.shg_pl_reader import SHGPLTransReader


def process_run(image_dictionary, image_analyser):

    print('process_run')
    print('entering for loop')

    reader = SHGPLTransReader()

    for prefix, data in image_dictionary.items():
        try:
            reader.assign_images(data)

            multi_image = reader.load_multi_image()

            image_analyser.image_analysis(
                multi_image, prefix)

            print('image_analysis done')
            # queue.put("Analysis of {} complete".format(prefix))

        except Exception as err:
            print('something went wrong')
            # queue.put("Error occurred in analysis of {}".format(prefix))
            raise err
