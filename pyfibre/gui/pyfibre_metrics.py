from pickle import UnpicklingError


class PyFibreMetrics:

    def __init__(self, parent, width=750, height=750):

        self.parent = parent
        self.width = width
        self.height = height

        self.window = Toplevel(self.parent.master)
        self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
        self.window.title('PyFibre - Metrics')
        self.window.geometry(f"{width}x{height}-100+40")

        self.frame = Frame(self.window)
        self.create_metrics()

    def create_metrics(self):

        self.metric_dict = {
            'No. Fibres': {"info": "Number of extracted fibres", "metric": IntVar(), "tag": "network"},
            'SHG Angle SDI': {"info": "Angle spectrum SDI of total image", "metric": DoubleVar(), "tag": "content"},
            'SHG Pixel Anisotropy': {"info": "Average anisotropy of all pixels in total image", "metric": DoubleVar(),
                                     "tag": "content"},
            'SHG Anisotropy': {"info": "Anisotropy of total image", "metric": DoubleVar(), "tag": "content"},
            'SHG Intensity Mean': {"info": "Average pixel intensity of total image", "metric": DoubleVar(),
                                   "tag": "content"},
            'SHG Intensity STD': {"info": "Pixel intensity standard deviation of total image", "metric": DoubleVar(),
                                  "tag": "content"},
            'SHG Intensity Entropy': {"info": "Average Shannon entropy of total image", "metric": DoubleVar(),
                                      "tag": "content"},
            'Fibre GLCM Contrast': {"info": "SHG GLCM angle-averaged contrast", "metric": DoubleVar(),
                                    "tag": "content"},
            'Fibre GLCM Homogeneity': {"info": "SHG GLCM angle-averaged homogeneity", "metric": DoubleVar(),
                                       "tag": "texture"},
            'Fibre GLCM Dissimilarity': {"info": "SHG GLCM angle-averaged dissimilarity", "metric": DoubleVar(),
                                         "tag": "texture"},
            'Fibre GLCM Correlation': {"info": "SHG GLCM angle-averaged correlation", "metric": DoubleVar(),
                                       "tag": "texture"},
            'Fibre GLCM Energy': {"info": "SHG GLCM angle-averaged energy", "metric": DoubleVar(), "tag": "texture"},
            'Fibre GLCM Similarity': {"info": "SHG GLCM angle-averaged similarity", "metric": DoubleVar(),
                                      "tag": "texture"},
            'Fibre GLCM Variance': {"info": "SHG GLCM angle-averaged variance", "metric": DoubleVar(),
                                    "tag": "texture"},
            'Fibre GLCM Cluster': {"info": "SHG GLCM angle-averaged clustering tendency", "metric": DoubleVar(),
                                   "tag": "texture"},
            'Fibre GLCM Entropy': {"info": "SHG GLCM angle-averaged entropy", "metric": DoubleVar(), "tag": "texture"},
            'Fibre GLCM Autocorrelation': {"info": "SHG GLCM angle-averaged autocorrelation", "metric": DoubleVar(),
                                           "tag": "texture"},
            'Fibre Area': {"info": "Average number of pixels covered by fibres", "metric": DoubleVar(),
                           "tag": "content"},
            'Fibre Coverage': {"info": "Ratio of image covered by fibres", "metric": DoubleVar(), "tag": "content"},
            'Fibre Linearity': {"info": "Average fibre segment linearity", "metric": DoubleVar(), "tag": "shape"},
            'Fibre Eccentricity': {"info": "Average fibre segment eccentricity", "metric": DoubleVar(), "tag": "shape"},
            'Fibre Density': {"info": "Average image fibre density", "metric": DoubleVar(), "tag": "texture"},
            'Fibre Hu Moment 1': {"info": "Average fibre segment Hu moment 1", "metric": DoubleVar(), "tag": "shape"},
            'Fibre Hu Moment 2': {"info": "Average fibre segment Hu moment 2", "metric": DoubleVar(), "tag": "shape"},
            'Fibre Waviness': {"info": "Average fibre waviness", "metric": DoubleVar(), "tag": "content"},
            'Fibre Lengths': {"info": "Average fibre pixel length", "metric": DoubleVar(), "tag": "content"},
            'Fibre Cross-Link Density': {"info": "Average cross-links per fibre", "metric": DoubleVar(),
                                         "tag": "content"},
            'Network Degree': {"info": "Average fibre network number of edges per node", "metric": DoubleVar(),
                               "tag": "network"},
            'Network Eigenvalue': {"info": "Max Eigenvalue of network", "metric": DoubleVar(), "tag": "network"},
            'Network Connectivity': {"info": "Average fibre network connectivity", "metric": DoubleVar(),
                                     "tag": "network"},

            'No. Cells': {"info": "Number of cell segments", "metric": IntVar(), "tag": "content"},
            'PL Angle SDI': {"info": "Angle spectrum SDI of total image", "metric": DoubleVar(), "tag": "content"},
            'PL Pixel Anisotropy': {"info": "Average anisotropy of all pixels in total image", "metric": DoubleVar(),
                                    "tag": "content"},
            'PL Anisotropy': {"info": "Anisotropy of total image", "metric": DoubleVar(), "tag": "content"},
            'PL Intensity Mean': {"info": "Average pixel intensity of total image", "metric": DoubleVar(),
                                  "tag": "content"},
            'PL Intensity STD': {"info": "Pixel intensity standard deviation of total image", "metric": DoubleVar(),
                                 "tag": "content"},
            'PL Intensity Entropy': {"info": "Average Shannon entropy of total image", "metric": DoubleVar(),
                                     "tag": "content"},
            'Cell GLCM Contrast': {"info": "PL GLCM angle-averaged contrast", "metric": DoubleVar(), "tag": "texture"},
            'Cell GLCM Homogeneity': {"info": "PL GLCM angle-averaged homogeneity", "metric": DoubleVar(),
                                      "tag": "texture"},
            'Cell GLCM Dissimilarity': {"info": "PL GLCM angle-averaged dissimilarity", "metric": DoubleVar(),
                                        "tag": "texture"},
            'Cell GLCM Correlation': {"info": "PL GLCM angle-averaged correlation", "metric": DoubleVar(),
                                      "tag": "texture"},
            'Cell GLCM Energy': {"info": "PL GLCM angle-averaged energy", "metric": DoubleVar(), "tag": "texture"},
            'Cell GLCM Similarity': {"info": "PL GLCM angle-averaged similarity", "metric": DoubleVar(),
                                     "tag": "texture"},
            'Cell GLCM Variance': {"info": "PL GLCM angle-averaged variance", "metric": DoubleVar(), "tag": "texture"},
            'Cell GLCM Cluster': {"info": "PL GLCM angle-averaged clustering tendency", "metric": DoubleVar(),
                                  "tag": "texture"},
            'Cell GLCM Entropy': {"info": "PL GLCM angle-averaged entropy", "metric": DoubleVar(), "tag": "texture"},
            'Cell GLCM Autocorrelation': {"info": "SHG GLCM angle-averaged autocorrelation", "metric": DoubleVar(),
                                          "tag": "texture"},
            'Muscle GLCM Contrast': {"info": "PL GLCM angle-averaged contrast", "metric": DoubleVar(),
                                     "tag": "texture"},
            'Muscle GLCM Homogeneity': {"info": "PL GLCM angle-averaged homogeneity", "metric": DoubleVar(),
                                        "tag": "texture"},
            'Muscle GLCM Dissimilarity': {"info": "PL GLCM angle-averaged dissimilarity", "metric": DoubleVar(),
                                          "tag": "texture"},
            'Muscle GLCM Correlation': {"info": "PL GLCM angle-averaged correlation", "metric": DoubleVar(),
                                        "tag": "texture"},
            'Muscle GLCM Energy': {"info": "PL GLCM angle-averaged energy", "metric": DoubleVar(), "tag": "texture"},
            'Muscle GLCM Similarity': {"info": "PL GLCM angle-averaged similarity", "metric": DoubleVar(),
                                       "tag": "texture"},
            'Muscle GLCM Variance': {"info": "PL GLCM angle-averaged variance", "metric": DoubleVar(),
                                     "tag": "texture"},
            'Muscle GLCM Cluster': {"info": "PL GLCM angle-averaged clustering tendency", "metric": DoubleVar(),
                                    "tag": "texture"},
            'Muscle GLCM Entropy': {"info": "PL GLCM angle-averaged entropy", "metric": DoubleVar(), "tag": "texture"},
            'Muscle GLCM Autocorrelation': {"info": "PL GLCM angle-averaged autocorrelation", "metric": DoubleVar(),
                                            "tag": "texture"},
            'Cell Area': {"info": "Average number of pixels covered by cells", "metric": DoubleVar(), "tag": "content"},
            'Cell Linearity': {"info": "Average cell segment linearity", "metric": DoubleVar(), "tag": "shape"},
            'Cell Coverage': {"info": "Ratio of image covered by cell", "metric": DoubleVar(), "tag": "content"},
            'Cell Eccentricity': {"info": "Average cell segment eccentricity", "metric": DoubleVar(), "tag": "shape"},
            'Cell Density': {"info": "Average image cell density", "metric": DoubleVar(), "tag": "texture"},
            'Cell Hu Moment 1': {"info": "Average cell segment Hu moment 1", "metric": DoubleVar(), "tag": "shape"},
            'Cell Hu Moment 2': {"info": "Average cell segment Hu moment 2", "metric": DoubleVar(), "tag": "shape"}
        }

        self.metric_titles = list(self.metric_dict.keys())

        self.metrics = [DoubleVar() for i in range(len(self.metric_titles))]
        self.headings = []
        self.info = []
        self.metrics = []

        self.texture = ttk.Labelframe(self.frame, text="Texture",
                                      width=self.width - 50, height=self.height - 50)
        self.content = ttk.Labelframe(self.frame, text="Content",
                                      width=self.width - 50, height=self.height - 50)
        self.shape = ttk.Labelframe(self.frame, text="Shape",
                                    width=self.width - 50, height=self.height - 50)
        self.network = ttk.Labelframe(self.frame, text="Network",
                                      width=self.width - 50, height=self.height - 50)

        self.frame_dict = {"texture": {'tab': self.texture, "count": 0},
                           "content": {'tab': self.content, "count": 0},
                           "shape": {'tab': self.shape, "count": 0},
                           "network": {'tab': self.network, "count": 0}}

        for i, metric in enumerate(self.metric_titles):
            tag = self.metric_dict[metric]["tag"]

            self.headings += [Label(self.frame_dict[tag]['tab'],
                                    text="{}:".format(metric), font=("Ariel", 8))]
            self.info += [Label(self.frame_dict[tag]['tab'],
                                text=self.metric_dict[metric]["info"], font=("Ariel", 8))]
            self.metrics += [Label(self.frame_dict[tag]['tab'],
                                   textvariable=self.metric_dict[metric]["metric"], font=("Ariel", 8))]

            self.headings[i].grid(column=0, row=self.frame_dict[tag]['count'])
            self.info[i].grid(column=1, row=self.frame_dict[tag]['count'])
            self.metrics[i].grid(column=2, row=self.frame_dict[tag]['count'])
            self.frame_dict[tag]['count'] += 1

        self.texture.grid(column=0, row=0, rowspan=3)
        self.content.grid(column=1, row=0)
        self.shape.grid(column=1, row=1)
        self.network.grid(column=1, row=2)

        self.frame.configure(background='#d8baa9')
        self.frame.pack()

    def get_metrics(self):

        selected_file = self.parent.file_display.tree.selection()[0]

        image_name = selected_file.split('/')[-1]
        image_path = '/'.join(selected_file.split('/')[:-1])
        fig_name = ut.check_file_name(image_name, extension='tif')
        data_dir = image_path + '/data/'

        try:
            loaded_metrics = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + fig_name)).iloc[0]
            for i, metric in enumerate(self.metric_dict.keys()):
                value = round(loaded_metrics[metric], 2)
                self.metric_dict[metric]["metric"].set(value)
            print("Displaying metrics for {}".format(fig_name))

        except (UnpicklingError, IOError, EOFError):
            print("Unable to display metrics for {}".format(fig_name))
            for i, metric in enumerate(self.metric_titles):
                self.metric_dict[metric]["metric"].set(0)

        self.parent.master.update_idletasks()

