class PyFibre():

    def __init__(self, master, n_proc=1):
        """Main PyFibre GUI

        Parameters
        ----------
        master: Tk
            Master Tk object
        n_proc: int
            Number of processors to run parallel analysis on

        """

        self.queue = Queue()
        self.processes = []

        self._initiate_variables()

        "Define GUI objects"
        self.master = master
        self.master.title(
            "PyFibre - Python Fibrous Image Analysis Toolkit"
        )
        self.master.geometry("700x720")
        self.master.configure(background=BACKGROUND_COLOUR)
        self.master.protocol("WM_DELETE_WINDOW", lambda: quit())

        self._create_frames()

        button_w = 15

        self.file_display = FileDisplay(self.master)

        self.file_display.run_button = Button(
            self.file_display, width=3 * button_w, text="GO",
            command=self.write_run)
        self.file_display.run_button.grid(column=0, row=2, columnspan=3)

        self.file_display.stop_button = Button(
            self.file_display, width=2 * button_w, text="STOP",
            command=self.stop_run, state=DISABLED)
        self.file_display.stop_button.grid(column=2, row=2, columnspan=2)

        self.file_display.progress = Progressbar(
            self.file_display, orient=HORIZONTAL, length=400,
            mode='determinate')
        self.file_display.progress.grid(column=0, row=3, columnspan=5)

        self.file_display.place(x=5, y=220, height=600, width=800)

        self.master.bind('<Double-1>', lambda e: self.update_windows())

    def _create_frames(self):

        self.title = Title(self.master, self.pyfibre_dir)
        self.title.place(
            bordermode=OUTSIDE, height=200, width=300
        )

        self.options = None
        self.toggle = Frame(self.master)
        self.toggle.configure(background=BACKGROUND_COLOUR)

        self.toggle.options_button = Button(
            self.toggle, width=15, text="Options",
            command=self._create_options)
        self.toggle.options_button.pack()

        self.viewer = None
        self.toggle.viewer_button = Button(
            self.toggle, width=15, text="Viewer",
            command=self._create_image_viewer)
        self.toggle.viewer_button.pack()

        self.metrics = None
        self.toggle.metric_button = Button(
            self.toggle, width=15, text="Metrics",
            command=self._create_metric_display)
        self.toggle.metric_button.pack()

        self.graphs = None
        self.toggle.graph_button = Button(
            self.toggle, width=15, text="Graphs",
            command=self._create_graph_display)
        self.toggle.graph_button.pack()

        self.toggle.test_button = Button(
            self.toggle, width=15, text="Test",
            command=self.test_image)
        self.toggle.test_button.pack()

        self.toggle.place(x=300, y=10, height=140, width=250)

    def _initiate_variables(self):

        "Initialise option variables"
        self.ow_metric = BooleanVar()
        self.ow_segment = BooleanVar()
        self.ow_network = BooleanVar()
        self.ow_figure = BooleanVar()
        self.save_db = BooleanVar()
        self.sigma = DoubleVar()
        self.sigma.set(0.5)
        self.p0 = IntVar()
        self.p0.set(1)
        self.p1 = IntVar()
        self.p1.set(99)
        self.n = IntVar()
        self.n.set(5)
        self.m = IntVar()
        self.m.set(35)
        self.alpha = DoubleVar()
        self.alpha.set(0.5)

        self.global_database = None
        self.fibre_database = None
        self.cell_database = None

    def _create_options(self):

        try:
            self.options.window.lift()
        except (TclError, AttributeError):
            self.options = PyFibreOptions(self)

    def _create_image_viewer(self):

        try:
            self.viewer.window.lift()
        except (TclError, AttributeError):
            self.viewer = PyFibreViewer(self)

    def _create_metric_display(self):

        try:
            self.metrics.window.lift()
        except (TclError, AttributeError):
            self.metrics = PyFibreMetrics(self)

    def _create_graph_display(self):

        try:
            self.graphs.window.lift()
        except (TclError, AttributeError):
            self.graphs = PyFibreGraphs(self)

    def update_windows(self):

        try:
            self.viewer.display_notebook()
        except (TclError, AttributeError):
            pass

        try:
            self.metrics.get_metrics()
        except (TclError, AttributeError):
            pass

        try:
            self.graphs.display_figures()
        except (TclError, AttributeError):
            pass

    def generate_db(self):

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        for i, input_file_name in enumerate(
                self.file_display.input_prefixes):

            image_name = input_file_name.split('/')[-1]
            image_path = '/'.join(input_file_name.split('/')[:-1])
            data_dir = image_path + '/data/'
            metric_name = data_dir + check_file_name(image_name, extension='tif')

            logger.info("Loading metrics for {}".format(metric_name))

            try:
                data_global = load_database(metric_name, '_global_metric')
                data_fibre = load_database(metric_name, '_fibre_metric')
                data_cell = load_database(metric_name, '_cell_metric')

                global_database = pd.concat([global_database, data_global], sort=True)
                fibre_database = pd.concat([fibre_database, data_fibre], sort=True)
                cell_database = pd.concat([cell_database, data_cell], sort=True)

            except (ValueError, IOError):
                logger.info(f"{input_file_name} databases not imported - skipping")

        self.global_database = global_database
        self.fibre_database = fibre_database
        self.cell_database = cell_database

        #self.update_dashboard()

    def save_database(self):

        db_filename = filedialog.asksaveasfilename()

        save_database(self.global_database, db_filename)
        save_database(self.fibre_database, db_filename, '_fibre')
        save_database(self.cell_database, db_filename, '_cell')

        self.update_log("Saving Database files {}".format(db_filename))

    def write_run(self):

        self.run_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        n_files = len(self.file_display.input_files)
        self.progress['maximum'] = n_files

        proc_count = np.min((self.n_proc, n_files))
        index_split = np.array_split(np.arange(n_files), proc_count)

        self.processes = []
        for indices in index_split:

            batch_files = [self.file_display.input_files[i] for i in indices]

            process = Process(target=image_analysis,
                    args=(batch_files,
                    (self.p0.get(), self.p1.get()),
                    (self.n.get(), self.m.get()),
                    self.sigma.get(), self.alpha.get(),
                    self.ow_metric.get(), self.ow_segment.get(),
                     self.ow_network.get(), self.ow_figure.get(),
                    self.queue))
            process.daemon = True
            self.processes.append(process)

        for process in self.processes:
            process.start()

        self.process_check()

    def process_check(self):
        """
        Check if there is something in the queue
        """
        self.queue_check()

        #if self.process.exitcode is None:
        if np.any([process.is_alive() for process in self.processes]):
            self.master.after(500, self.process_check)
        else:
            self.stop_run()
            self.generate_db()
            if self.save_db.get(): self.save_database()

    def queue_check(self):

        while not self.queue.empty():
            try:
                msg = self.queue.get(0)
                self.update_log(msg)
                self.progress.configure(value=self.progress['value'] + 1)
                self.progress.update()
            except queue.Empty: pass

    def stop_run(self):

        self.update_log("Stopping Analysis")
        for process in self.processes:
            process.terminate()
        self.progress['value'] = 0
        self.run_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)


    def update_log(self, text):

        self.Log += text + '\n'

    def test_image(self):

        if self.file_display['state'] == NORMAL:
            input_files = [self.pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif']

            self.file_display.add_files(input_files)

            self.run_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
            self.progress['maximum'] = len(input_files)

            proc_count = np.min((self.n_proc, len(input_files)))
            index_split = np.array_split(np.arange(len(input_files)), proc_count)

            self.processes = []
            for indices in index_split:

                batch_files = [input_files[i] for i in indices]

                process = Process(target=run_analysis,
                        args=(batch_files,
                        (self.p0.get(), self.p1.get()),
                        (self.n.get(), self.m.get()),
                        self.sigma.get(), self.alpha.get(),
                        self.ow_metric.get(), self.ow_segment.get(),
                         self.ow_network.get(), self.ow_figure.get(),
                        self.queue))
                process.daemon = True
                self.processes.append(process)

            for process in self.processes: process.start()

            self.process_check()
