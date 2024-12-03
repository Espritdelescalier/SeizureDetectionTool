import os
import sys
import traceback
import configparser
import importlib
import datetime
import csv
import re
import bisect
import math
import os.path as path
import numpy as np
import matplotlib
import matplotlib.widgets as mwidgets
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import (QMainWindow, QWidget,
                               QLineEdit, QToolBar, QStatusBar, QFileDialog,
                               QScrollArea, QFrame, QTabWidget, QBoxLayout,
                               QVBoxLayout, QPushButton, QLayout, QCheckBox, QGridLayout, QLabel)
from PySide6.QtCore import (QObject, QThreadPool, Slot, Signal,
                            QTimer, QRunnable, QPropertyAnimation,
                            QSequentialAnimationGroup, QPoint, Qt)
from PySide6 import QtGui
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget
from functools import partial
from extracttrace import decompose_signal, findCandidates, traces_info_from_file
from signalcandidate import SignalCandidate, MetaAndDataFile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pyedflib
import pyabf

matplotlib.use('QtAgg')


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup,
    signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
    Supplied args and kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class CandidatesWindow(QWidget):

    def __init__(self, indices_to_display, all_candidates, displayAno):
        super().__init__()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.inner = QFrame(self.scrollArea)
        self.inner.setLayout(QVBoxLayout())
        self.scrollArea.setWidget(self.inner)
        display_button = []
        count = 0
        print(indices_to_display)
        for i in indices_to_display:
            print(i)
            display_button.append(
                QPushButton(
                    "{} {}".format(i + 1, all_candidates[i].capture_modality),
                    self
                )
            )
            display_button[count].released.connect(
                partial(displayAno, i))
            self.inner.layout().addWidget(display_button[count])
            count += 1


class popup(QWidget):
    def __init__(self, parent_geometry):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Dialog)
        self.setWindowTitle("Processing")
        self.width = 200
        self.height = 200
        self.move(parent_geometry.x()-self.width /
                  2, parent_geometry.y()-self.height/2)
        self.resize(self.width, self.height)
        self.child = []
        self.anim_p = []
        self.anim_p2 = []
        self.anim_group = QSequentialAnimationGroup()
        for i in range(11):
            self.child.append(QWidget(self))
            self.child[i].move(50+i*10, 100+math.sin(i*0.7+15)*10)
            self.child[i].setStyleSheet(
                "background-color:darkred;border-radius:15px;")
            self.child[i].resize(5, 10)
            self.anim_p.append(QPropertyAnimation(self.child[i], b"pos"))
            self.anim_p[i].setEndValue(
                QPoint(50+i*10, 100+math.cos(i*0.7+15)*10))
            self.anim_p[i].setDuration(50)
            self.anim_p2.append(QPropertyAnimation(self.child[i], b"pos"))
            self.anim_p2[i].setEndValue(
                QPoint(50+i*10, 100+math.sin(i*0.7+15)*10))
            self.anim_p2[i].setDuration(50)
            self.anim_group.addAnimation(self.anim_p[i])
            self.anim_group.addAnimation(self.anim_p2[i])

        # self.start()
        timer = QTimer(self)
        timer.setInterval(505)  # interval in ms
        timer.timeout.connect(self.start)
        timer.start()

    def start(self):
        self.anim_group.start()

    def closeEvent(self, event):
        event.ignore()

    def update_text(self, text):
        # TODO message processing please wait
        self.label.setText(text)
        self.label.setWordWrap(True)
        self.label.adjustSize()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel('mV')
        self.axes.set_xlabel('time (ms)')
        ###########
        fig.tight_layout()
        ###########
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):

    def __init__(self, *args, title="EEG Anomaly Detection", **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Definition of the window title (default title otherwise)
        self.setWindowTitle(title)

        self.sc = MplCanvas(self, width=12, height=6, dpi=100)
        self.sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
        self.sc.mpl_connect('button_press_event', self.on_mouse_pos_keypress)

        self.ft = MplCanvas(self, width=12, height=3, dpi=100)
        self.ft.mpl_connect('motion_notify_event', self.on_mouse_move_display)
        self.ft.mpl_connect('button_press_event',
                            self.on_dblclick_select_timeline)

        # initialization of the grid for the options to be checked
        # by the user in the interface parameters
        self.grid_layout_tick_param = QGridLayout()
        self.num_subsequences = -1
        self.tickboxes = []
        self.checked_indices = []

        self.wtmra = None
        self.smplrate = 100
        self.factor = 100
        # for size of signal loaded on user display
        # block_size will be used to divide the entire signal into
        # fixed-sized blocks to load only parts of the signal into memory
        self.block_size = 10000
        self.blocks = {}

        # for saving data in formats
        self.abf = ""
        self.edf = ""

        self.allCandidates = []
        self.metaFile = None
        self.fullTrace = None
        self.tailHeadAno = 0
        self.traceProcessingDone = False
        self.currentOpenedAno = -1
        self.previousOpenedAno = -1
        self.anoCount = 0
        self.newMinMax = None
        self.selectedAno = []
        self.background = False
        self.samplerate = 0
        self.dataLength = 0
        self.nbOfSelected = 0
        self.addedAno = 0
        self.max_trace = 0
        self.min_trace = 0
        self.savedconfig = False
        self.selected_channels = 0
        self.popup = None
        self.filePath = ""
        self.saveFolder = "/data"
        self.fullCSVPath = ""
        self.anim_file_name = None
        self.played_gif = False
        self.loading = None
        self.threadpool = QThreadPool()

        self.anoWindows = {
            "all": None,
            "selected": None
        }
        """
        Load layout orientation from config file
        """
        self.layout_directions = {
            'vertical': QBoxLayout.TopToBottom,
            'horizontal': QBoxLayout.LeftToRight
        }
        """
        Open config file
        """
        self.config = configparser.ConfigParser()
        pathfile = os.path.abspath(__file__).split('/')[:-1]
        joinseq = '/'
        path_file_join = joinseq.join(pathfile)
        full_config_path = os.path.join(path_file_join, "config.ini")
        self.config.read(full_config_path)
        """
        Load default algorithm values and button types from config file
        """
        self.allvalues = self.config['DEFAULT_VALUES']['val'].split(',')
        self.button_types = self.config['button_types']['types'].split('\n')
        """
        Load tabs and layouts
        """
        self.tab_names = self.config['tabs']['tab_names'].split('\n')
        self.layout_names = self.config['layout']['names'].split('\n')
        self.layout_orientation = self.config['layout']['orientation'].split(
            '\n')
        """
        Load text box default text and description from config file
        """
        self.button_text = self.config['text_boxes']['boxes'].split('\n')
        self.description_button = self.config['text_boxes'][
            'descriptions'].split('\n')
        """
        Connect text box to input values
        """
        ####
        self.text_box()
        # self.affiche_nb_decomposition()
        self.add_multi_select_subsequences_tick()

        """
        Instantiate all button dicts to load:
        button names (class names from Qt)
        button descriptions
        button shortcuts
        Instantiate dict containing a list with all
        Qt button objects
        """
        self.button_names = dict()
        self.button_desc = dict()
        self.button_sc = dict()
        self.all_buttons = dict()

        for butt in self.button_types:
            self.all_buttons[butt] = []
        """
        Load names, descriptions and shortcuts for each button types
        Qaction, QPushButton, QCheckBox
        """
        for btype in self.button_types:
            self.button_names[btype] = self.config[btype]['names'].split('\n')
            self.button_desc[btype] = self.config[btype]['descriptions'].split(
                '\n')
            self.button_sc[btype] = self.config[btype]['shortcuts'].split('\n')
        """
        Main toolbar for opening files and choosing saving folder
        """
        self.mainToolbar = QToolBar("The main toolbar")
        self.addToolBar(self.mainToolbar)
        """
        Create buttons for main toolbar
        """
        self.type_cycle_button_create()
        """
        Connect each button to its function
        """
        self.all_buttons['QAction'][0].triggered.connect(
            partial(self.open_file_button_clicked, "(*.abf *.edf *.edf+)"))
        self.all_buttons['QAction'][0].triggered.connect(
            self.add_multi_select_subsequences_tick)
        self.all_buttons['QAction'][1].triggered.connect(
            self.save_folder_button_clicked)
        self.all_buttons['QAction'][2].triggered.connect(
            partial(self.open_config_button_clicked, "(*.config)"))
        self.all_buttons['QAction'][3].triggered.connect(
            partial(self.save_parameters_config_file, ""))
        self.all_buttons['QAction'][4].triggered.connect(
            lambda x: self.show_new_window(
                list(range(len(self.get_all_candidates()))),
                "all"
            )
        )
        self.all_buttons['QAction'][5].triggered.connect(
            lambda x: self.show_new_window(
                np.array(self.get_selected_ano()).nonzero()[0],
                "selected"
            )
        )

        """Boutons pour actions spécifiques"""

        self.all_buttons['QPushButton'][0].released.connect(
            partial(self.display_step, -1))
        self.all_buttons['QPushButton'][1].released.connect(
            self.display_popup)
        self.all_buttons['QPushButton'][1].released.connect(
            self.get_checked_indices)
        self.all_buttons['QPushButton'][1].released.connect(self.launch_thread)
        self.all_buttons['QPushButton'][2].released.connect(self.save_done)
        self.all_buttons['QPushButton'][3].released.connect(
            partial(self.display_step, 1))
        self.all_buttons['QCheckBox'][0].released.connect(self.check_uncheck)
        self.all_buttons['QCheckBox'][1].released.connect(
            self.background_check)

        self.add_action_toolbar('QAction')
        self.connect_combobox()

        self.setStatusBar(QStatusBar(self))
        """
        Create toolbar, passing canvas as first parameter,
        parent (self, the MainWindow) as second.
        """
        toolbar = NavigationToolbar(self.sc, self)
        """
        Instantiate all layouts with orientation as parameter
        """
        self.all_layouts = dict()
        count = 0
        for i in (self.layout_names):
            self.all_layouts[i] = QBoxLayout(
                self.layout_directions[self.layout_orientation[count]])
            self.all_layouts[i].setSpacing(20)
            count += 1
        # QVBoxLayout() and QHBoxLayout() equivalent but with orientation
        # taken from config file
        """
        Grid for options to be checked by the user in the interface parameters
        """
        # menu tickboxes for the options to be checked by the user in the interface parameters
        self.all_layouts['parameters'].addLayout(self.grid_layout_tick_param)
        """
        All Qline in parameters layout
        """

        for i in range(len(self.description_button)):
            self.all_layouts['parameters'].addWidget(self.textedit[i])

        """
        Organise each button widget into its corresponding layout
        """
        self.widget_to_layout('QPushButton', 'control')
        self.widget_to_layout('QCheckBox', 'selection')
        self.widget_to_layout('QComboBox', 'selection')
        self.widget_to_layout('QLabel', 'label')

        self.all_layouts['control'].setAlignment(QtCore.Qt.AlignCenter)
        self.all_layouts['selection'].setAlignment(QtCore.Qt.AlignCenter)
        self.all_layouts['label'].setAlignment(QtCore.Qt.AlignLeft)
        """
        Organise all previous layouts
        the toolbar, and both matplotlib viewports
        into the main layout mlayout
        """
        self.all_layouts['con_selct'].addLayout(self.all_layouts['control'])
        self.all_layouts['con_selct'].addLayout(self.all_layouts['selection'])
        self.all_layouts['info_interact'].addLayout(self.all_layouts['label'])
        self.all_layouts['info_interact'].addLayout(
            self.all_layouts['con_selct'])
        self.all_layouts['info_interact'].setSizeConstraint(
            QLayout.SetFixedSize)
        # self.all_layouts['mlayout'].addLayout(self.all_layouts['control'])
        # self.all_layouts['mlayout'].addLayout(self.all_layouts['selection'])
        self.all_layouts['mlayout'].addLayout(
            self.all_layouts['info_interact'])
        self.all_layouts['mlayout'].addWidget(toolbar)
        self.all_layouts['mlayout'].addWidget(self.sc)
        self.all_layouts['mlayout'].addWidget(self.ft)
        """
        widgetParam contains parameters layout
        """
        widgetParam = QWidget()
        widgetParam.setLayout(self.all_layouts['parameters'])
        """
        widget contains main layout
        """
        widget = QWidget()
        widget.setLayout(self.all_layouts['mlayout'])
        """
        Add each high level widget into corresponding tab
        """
        self.tab_names = self.config['tabs']['tab_names'].split('\n')

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(True)
        tabs.addTab(widget, self.tab_names[0])
        tabs.addTab(widgetParam, self.tab_names[1])

        self.setCentralWidget(tabs)

        self.span = mwidgets.SpanSelector(self.sc.axes,
                                          self.onselect,
                                          "horizontal",
                                          useblit=True,
                                          props=dict(alpha=0.2,
                                                     facecolor="orange"),
                                          interactive=True,
                                          drag_from_anywhere=True)
        xmin, xmax = 1, 3
        self.span.extents = (xmin, xmax)
        self.span.onselect(xmin, xmax)

        self.show()

    def get_all_candidates(self):
        return self.allCandidates

    def get_selected_ano(self):
        return self.selectedAno

    def widget_to_layout(self, btype, layout_name):
        for i in range(len(self.all_buttons[btype])):
            self.all_layouts[layout_name].addWidget(self.all_buttons[btype][i])

    def type_cycle_button_create(self):
        for btypes in self.button_types:
            self.create_button(btypes)

    def get_adequate_module(self, buttonname):
        my_module = None
        if re.match("QAction", buttonname):
            my_module = importlib.import_module("PySide6.QtGui")
        else:
            my_module = importlib.import_module("PySide6.QtWidgets")

        return my_module

    def create_button(self, btype):
        for i in range(len(self.button_names[btype])):
            print(btype)
            my_module = self.get_adequate_module(btype)
            relevant_button_class = getattr(my_module, btype)
            # args = dir(relevant_button_class)
            # print(args, '\n')
            if btype == 'QComboBox':
                self.all_buttons[btype].append(relevant_button_class(self))
            elif btype == 'QLabel':
                self.all_buttons[btype].append(
                    relevant_button_class(self.button_names[btype][i], self))
                self.all_buttons[btype][i].setWordWrap(True)
                self.all_buttons[btype][i].setAlignment(QtCore.Qt.AlignLeft)
                self.all_buttons[btype][i].adjustSize()
            else:
                self.all_buttons[btype].append(
                    relevant_button_class(self.button_names[btype][i], self))
                self.all_buttons[btype][i].setShortcut(
                    QtGui.QKeySequence(self.button_sc[btype][i]))

            self.all_buttons[btype][i].setStatusTip(self.button_desc[btype][i])

    def add_action_toolbar(self, btype):
        for i in range(len(self.button_names[btype])):
            self.mainToolbar.addAction(self.all_buttons[btype][i])

    """
    Connect combobox to function
    """

    def connect_combobox(self):
        for c in range(len(self.all_buttons['QComboBox'])):
            self.all_buttons['QComboBox'][c].activated.connect(
                partial(self.combobox_select, c))

    def combobox_select(self, index, john):
        channel = self.all_buttons['QComboBox'][index].currentText()
        print(channel)
        self.selected_channels = self.metaFile.channels_labels.index(channel)

    #####################################################################################

    """
    Analysis of frequencies contained in subsequence using the FFT.
    Return: the frequency range of a subsequence.
    """
    # TF : represent in frequencies (and amplitudes) non-periodic signals (represented as a function of time)

    def get_minMax_frequency(self, subsequence, sample_rate):
        """Fourier Transform (FT)"""
        # -> table of complex coeffs which represent amplitudes and phases of the sinusoids of the signal
        # ret : does not directly give the frequencies of the signal, but a representation of the signal in the frequency domain
        # nb complexes :
        #   - real part: amplitude of the cos component
        #   - imaginary part: amplitude of the sin component
        ft_vals = np.fft.fft(subsequence)  # signal amplitude spectrum

        # to obtain the frequencies: calculation of the amplitudes of the coeffs and associate them with the corresponding frequencies
        # ret : tab containing sampling frequencies
        # 1/sample_rate : time interval between samples (inverse of sample rate)
        frequencies = np.fft.fftfreq(len(subsequence), 1/sample_rate)

        # to find the amplitude associated with each complex coeff (each frequency), we use the module(=np.abs) |z| = sqrt(partRel^2 + partIma^2)
        # -> amplitude of sinusoids for each frequency
        # note : higher values ​​of 'amplitudes' indicate the most important frequencies of the signal
        amplitudes = np.abs(ft_vals)

        """Filtering positive frequencies"""
        # TF : symmetrical
        # we keep only the first half of the frequencies/amplitudes (the positive ones)
        positive_frequencies = frequencies[:len(frequencies)//2]
        positive_amplitudes = amplitudes[:len(amplitudes)//2]

        """Filtering of significant frequencies"""
        # 90th percentile to focus on
        # the most powerful 10% and eliminate background noise
        # only retains important frequencies
        threshold = np.percentile(positive_amplitudes, 90)
        # extracts frequencies with large/significant amplitude
        significant_frequencies = positive_frequencies[positive_amplitudes > threshold]
        # if no significant frequency
        if len(significant_frequencies) == 0:
            return 0, 0

        # we take the smallest and largest frequency among the filtered frequencies
        min_freq = np.min(significant_frequencies)
        max_freq = np.max(significant_frequencies)

        """Show values"""
        # print("ft_vals:\n", ft_vals)
        # print("\nfrequencies:\n", frequencies)
        # print("\namplitudes:\n", amplitudes)
        # print("\nthreshold:\n", threshold)
        # print("\nsignificant_frequencies:\n", significant_frequencies)

        """Display frequency/amplitude graphs (Fourrier)"""
        # plt.figure(figsize=(12, 6))
        # plt.plot(positive_frequencies, positive_amplitudes)
        # plt.title("Frequency spectrum")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()
        return min_freq, max_freq

    def add_multi_select_subsequences_tick(self):
        self.setLayout(self.grid_layout_tick_param)

        # if you want more columns, just change nbColumns in config.ini
        nb_col = int(self.config['DEFAULT_VALUES']
                     ['nb_columns_tickboxes_parameters'])

        # if title already exists, delete it (the function is called several times)
        if hasattr(self, 'title') and self.title:
            self.grid_layout_tick_param.removeWidget(self.title)
            self.title.deleteLater()
        # title
        self.title = QLabel(
            "Select the subsequences to analyse: \nFrom last to first the frequency ranges are successively divided by two from the original frequency of the signal", self)
        self.grid_layout_tick_param.addWidget(
            self.title, 0, 0, 1, nb_col, Qt.AlignmentFlag.AlignCenter)

        # remove all existing checkboxes
        for widget in self.tickboxes:
            widget.setParent(None)
            widget.deleteLater()
        self.tickboxes = []

        # nb default options when no file open
        nb_elements = self.num_subsequences if self.num_subsequences != - \
            1 else int(self.config['DEFAULT_VALUES']['default_tickboxes_parameters'])

        options_per_column = nb_elements // nb_col
        modulo = nb_elements % nb_col
        # create checkboxes
        if nb_elements != 0:
            current_index = 0
            for col in range(nb_col):
                column_tickboxes = options_per_column + \
                    (1 if modulo > 0 else 0)
                modulo -= 1

                for row in range(column_tickboxes):
                    if self.wtmra is not None and current_index < len(self.wtmra):
                        # give the frequency ranges of each subsequence
                        subseq = self.wtmra[current_index]
                        min_freq, max_freq = self.get_minMax_frequency(
                            subseq, self.smplrate)
                        freq_minMax = f"{min_freq:.1f}-{max_freq:.1f} Hz"
                        tickbox_txt = f"subsequence {current_index + 1}"
                    else:
                        tickbox_txt = f"subsequence {current_index + 1}"

                    tickbox = QCheckBox(tickbox_txt, self)
                    if col == 0:
                        alignment = Qt.AlignmentFlag.AlignLeft
                    elif col == nb_col - 1:
                        alignment = Qt.AlignmentFlag.AlignRight
                    else:
                        alignment = Qt.AlignmentFlag.AlignCenter
                    self.grid_layout_tick_param.addWidget(
                        tickbox, row + 1, col, alignment)
                    self.tickboxes.append(tickbox)
                    current_index += 1

        # check 2 (depends on config.ini) boxes by default
        default_tickboxes_nb = int(
            self.config['DEFAULT_VALUES']['default_tickboxes_parameters'])
        for i in range(min(default_tickboxes_nb, len(self.tickboxes))):
            self.tickboxes[i].setChecked(True)

        self.show()
        return self.tickboxes

    def get_checked_indices(self):
        for index, tickbox in enumerate(self.tickboxes):
            if tickbox.isChecked():
                self.checked_indices.append(index)
        print(f"Checked indices: {self.checked_indices}")
        return self.checked_indices

    def affiche_nb_decomposition(self):
        if self.metaFile and self.selected_channels is not None:
            # short sequence
            trace = self.metaFile.full_signal[self.selected_channels][:20000]
            sample_rate = int(
                self.metaFile.sample_rate[self.selected_channels])
            decompLevel = int(self.allvalues[1])
            factor = int(self.allvalues[-1])
            wavelet = self.allvalues[-2]
            #tup = (trace, sample_rate, decompLevel, factor, wavelet)
            try:
                # th = ThreadWithReturnValue(target=decompose_signal,
                #                                   args=(*tup, ))
                # th.start()
                #result = th.join()
                _, self.wtmra, _, _, self.smplrate, self.num_subsequences = decompose_signal(
                    trace, sample_rate, decompLevel, factor, wavelet)
                print(f"Number of subsequences : {self.num_subsequences}")
            except Exception as e:
                print(f"Error in decompose_signal: {e}")
        print("End of the function affiche_nb_decomposition")

    #####################################################################################

    """
    Text boxes instantiation from config file
    """

    def text_box(self):
        num = len(self.button_text)
        self.textedit = []
        for i in range(num):
            self.textedit.append(QLineEdit())
            self.textedit[i].setMaxLength(10)
            self.textedit[i].setPlaceholderText(self.button_text[i])
            self.textedit[i].setStatusTip(self.description_button[i])
            self.textedit[i].returnPressed.connect(
                partial(self.return_pressed, i))

    def update_placeholder_text(self, string):
        for i in range(len(self.textedit)):
            self.textedit[i].setPlaceholderText(
                str(self.allvalues[i]) + string)
    """
    Updates text/formatting of a label
    in the user interface
    """

    def update_label(self, index, string):
        self.all_buttons['QLabel'][index].setText(string)
        self.all_buttons['QLabel'][index].setWordWrap(True)
        self.all_buttons['QLabel'][index].adjustSize()
        self.all_buttons['QLabel'][index].setAlignment(QtCore.Qt.AlignLeft)

    def update_combobox(self, index, drop_down_list):
        self.all_buttons['QComboBox'][index].clear()
        self.all_buttons['QComboBox'][index].addItems(drop_down_list)

    """
    Function called upon selection with span on matplotlib plot
    """

    def onselect(self, min_value, max_value):
        if self.currentOpenedAno >= 0:
            self.newMinMax[self.currentOpenedAno] = [
                int(min_value), int(max_value)
            ]
        return int(min_value), int(max_value)

    """
    Function called upon pressing enter in each field, will add the value
    in each field in a list
    """

    def return_pressed(self, index):
        text = self.textedit[index].text()
        self.allvalues[index] = text
        # dynamic change of the sequence selection menu depending on the decomposition selected by the user
        if index == 1 and text and self.metaFile:
            self.affiche_nb_decomposition()
            self.add_multi_select_subsequences_tick()

    def tr(self, text):
        return QObject.tr(text)

    """
    Function called by using the open file button
    """

    def open_file_dialog(self, s):
        path = QFileDialog.getOpenFileName(self,
                                           self.tr("Open file"),
                                           self.tr("~/"),
                                           self.tr(s))
        return path

    def open_file_button_clicked(self, s):
        self.filePath, _ = self.open_file_dialog(s)
        print("filepath from dialogbox", len(self.filePath))
        if self.filePath is not None and len(self.filePath) != 0:
            # to retrieve later, meta data of the initial abf (or edf) file -> newfiles.edf
            if self.filePath.lower().endswith('.edf'):
                self.edf = pyedflib.EdfReader(self.filePath)
                self.edf.close()
            elif self.filePath.lower().endswith('.abf'):
                self.abf = pyabf.ABF(self.filePath)
            #
            tup = traces_info_from_file(self.filePath)
            self.metaFile = MetaAndDataFile(*tup)
            self.savedconfig = False
            user_info = 'File ' + self.filePath.split('/')[-1] + ' opened'
            self.update_label(0, user_info)
            self.selected_channels = self.metaFile.eeg_channel_index
            self.update_combobox(0, self.metaFile.channels_labels)

            self.affiche_nb_decomposition()

    def open_config_button_clicked(self, s):
        filePath, _ = self.open_file_dialog(s)
        if filePath is not None:
            self.load_params(filePath)
            self.update_placeholder_text(" parameter loaded from file " +
                                         filePath.split("/")[-1])

    def load_params(self, filepath):
        self.config = configparser.ConfigParser()
        self.config.read(filepath)
        self.allvalues = self.config['parameters']['values'].split('\n')
        print(self.allvalues)

    def save_folder_button_clicked(self, s):
        self.saveFolder = QFileDialog.getExistingDirectory(
            self, self.tr("Save folder"), self.tr("~/"))
        ####
        if self.saveFolder:
            print(f"Save folder selected: {self.saveFolder}")
        else:
            print("No folder selected")
        ####

    def save_parameters_config_file(self, string):
        # check if file has been opened
        if self.filePath == "":
            tempnamefile = string + "parameterstemp"
        else:
            tempnamefile = self.filePath.split("/")[-1]
        # get filepath from dialog box
        paramfile, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save parameters config"),
            self.tr(tempnamefile[:-4] + ".config"))
        self.save_config(paramfile)

        if paramfile != "":
            self.savedconfig = True

    def save_config(self, filename):
        config = configparser.ConfigParser()
        stringvalues = ""
        for i in self.allvalues:
            stringvalues = stringvalues + str(i) + '\n'
        config['parameters'] = {'values': stringvalues}
        if filename != "":
            with open(filename, 'w') as configfile:
                config.write(configfile)

    def show_new_window(self, indices, name):
        if self.anoWindows[name] is None:
            self.anoWindows[name] = CandidatesWindow(
                indices, self.allCandidates, self.displayAno)
            self.anoWindows[name].scrollArea.show()
            print("window is none")
        elif not self.anoWindows[name].isVisible():
            print("window is not visible")
            self.anoWindows[name].scrollArea.close()
            self.anoWindows[name].close()
            self.anoWindows[name] = None
            self.anoWindows[name] = CandidatesWindow(
                indices, self.allCandidates, self.displayAno)
            self.anoWindows[name].scrollArea.show()
        else:
            print("window is in limbo")
            self.anoWindows[name].scrollArea.close()
            self.anoWindows[name].close()
            self.anoWindows[name] = None

    def find_ano(self):
        print("vroom vroom I find anomalies")
        user_info = 'Processing has started on file ' +\
            self.filePath.split('/')[-1]
        self.update_label(2, user_info)
        if self.traceProcessingDone is False:
            self.traceProcessingDone = True
        else:
            self.candidatesWindow = None
            self.selectedCandidatesWindow = None
            self.fullTrace = None
            self.tailHeadAno = 0
            self.currentOpenedAno = -1
            self.nbOfSelected = 0
            self.addedAno = 0
            self.sc.axes.clear()
        tup = (self.metaFile.full_signal[self.selected_channels],
               int(self.metaFile.sample_rate[self.selected_channels]),
               int(self.allvalues[0]), int(self.allvalues[1]),
               int(self.allvalues[2]), self.checked_indices,
               float(self.allvalues[5]), int(self.allvalues[6]),
               False, self.allvalues[8], int(self.allvalues[9]))

        th = findCandidates(*tup)

        # th.start()
        user_info = 'Processing has started on file ' +\
            self.filePath.split('/')[-1]
        self.update_label(2, user_info)
        # self.all_buttons['QLabel'][2].repaint()

        self.fullTrace, self.allCandidates,\
            self.tailHeadAno, self.samplerate, self.num_subsequences = th  # th.join()
        # self.loading.inner.close()

        self.anoCount = len(self.allCandidates)
        self.newMinMax = [[0, 0]] * self.anoCount
        self.selectedAno = [0] * self.anoCount
        self.min_trace = min(self.fullTrace[:])
        self.max_trace = max(self.fullTrace[:])

        self.ft_update_display()
        self.ft.draw()

        self.sc.axes.plot(self.fullTrace[:],
                          marker='|',
                          markerfacecolor='r',
                          markeredgecolor='r',
                          markevery=[x.onset for x in self.allCandidates],
                          markersize=2000,
                          linewidth=0.5,
                          color='b')
        user_info = 'Processing on file ' +\
            self.filePath.split('/')[-1] + ' has concluded'
        self.update_label(2, user_info)

    def thread_complete(self):
        user_info = 'Processing done on file ' +\
            self.filePath.split('/')[-1]
        self.update_label(2, user_info)
        self.loading.close()
        self.loading = None

    def print_output(self, s):
        print("output", s)

    def launch_thread(self):
        worker = Worker(self.find_ano)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # worker.signals.progress.connect(self.progress_processing)

        self.threadpool.start(worker)

    def display_popup(self):
        if self.loading is None:
            self.loading = popup(self.geometry().center())
            # self.loading.inner.show()
            self.loading.show()
            print("window is none")
        elif not self.loading.isVisible():
            print("window is not visible")
            self.loading.show()

    ##########################################
    def load_signal_segment(self, start, end):
        """Charge un segment de signal en mémoire"""
        # Each block = a part of the signal/sequence
        # we divide the signal/sequence into blocks to facilitate loading parts of the signal into memory
        # (so as not to load the entire signal on each display)

        # Calculates start index of the block which contains the starting point of the desired signal segment
        block_start = (start // self.block_size) * self.block_size
        # Calculates end index of the block that contains the end point of the signal segment
        # + 1 to include the block with the end point
        block_end = ((end // self.block_size) + 1) * self.block_size

        # loop going from start to end with a step of block_size
        for block in range(block_start, block_end, self.block_size):
            if block not in self.blocks:
                print(
                    f"Loading block from {block} to {block + self.block_size}")
                # load/stock current block in memory from fuulTrace
                # dico: keys (block) = start indices of each block; values ​​= signal data
                # self.fullTrace[indStartBlock:indEndBlock]
                self.blocks[block] = self.fullTrace[block:block +
                                                    self.block_size]

        # segment for display by merging loaded blocks
        segment = np.concatenate([self.blocks[block] for block in range(
            block_start, block_end, self.block_size)])
        print(
            f"Segment loaded from {start} to {end}, size: {segment.nbytes} bytes")
        # segment starts and ends at the correct indices (start and end)
        return segment[start - block_start:end - block_start]
    ##########################################

    def signal_size(self, signal):
        # Returns the size in bytes of the signal
        return signal.nbytes

    def displayAno(self, index):
        self.previousOpenedAno = self.currentOpenedAno
        self.currentOpenedAno = index
        if self.selectedAno[index] == 0:
            self.all_buttons['QCheckBox'][0].setChecked(False)
            self.all_buttons['QCheckBox'][0].repaint()
        else:
            self.all_buttons['QCheckBox'][0].setChecked(True)
            self.all_buttons['QCheckBox'][0].repaint()
        self.debug_display(index)
        self.sc.axes.xaxis.set_major_locator(mticker.MultipleLocator(1000))
        self.sc.axes.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

        # Full signal size
        total_signal_size = self.signal_size(self.fullTrace)
        print(f"Total signal size: {total_signal_size} bytes")

        """
        zoom in on the x axis with the display param
        """
        # Adjusts the x-axis limits to zoom in on the ano
        # forcing int
        newxlone = int(self.allCandidates[index].onset -
                       int(self.allvalues[7]) * self.samplerate)
        newxltwo = int(self.allCandidates[index].end +
                       int(self.allvalues[7]) * self.samplerate)
        self.sc.axes.set_xlim(newxlone, newxltwo)
        """
        zoom in on the y axis with the amplitude of the candidate's signal
        """
        # Loads only the necessary segment of the signal
        segment = self.load_signal_segment(newxlone, newxltwo)
        newylone = max(segment)
        newyltwo = min(segment)
        """
        scale the signal in the Y-axis to +/- a 10th of the amplitude
        of the signal displayed
        """
        amp = newylone - newyltwo
        factor = amp * 0.1
        self.sc.axes.set_ylim(newyltwo - factor, newylone + factor)
        """
        change the ticks in each axis and the associated legends
        """
        # Update axis legends with new scales

        a = self.sc.axes.get_xticks().tolist()
        self.sc.axes.xaxis.set_major_locator(mticker.FixedLocator(a))
        newticks = [str(float(x) / (self.samplerate / 1000)) for x in a]
        self.sc.axes.set_xticklabels(newticks)

        self.sc.axes.set_ylabel('mV')
        self.sc.axes.set_xlabel('time (ms)')
        """
        compute and display span select
        """
        xmin = (self.newMinMax[index][0],
                self.allCandidates[index].onset)[self.selectedAno[index] == 0]
        xmax = (self.newMinMax[index][1],
                self.allCandidates[index].end)[self.selectedAno[index] == 0]

        self.sc.axes.yaxis.set_major_locator(mticker.AutoLocator())

        self.span_add(xmin, xmax)
        self.sc.draw()

        self.ft_update_display()

        # keep green verticle bar after decreasing the resolution
        downsampled_onset = self.allCandidates[self.currentOpenedAno].onset // self.factor
        if 0 <= downsampled_onset < len(self.fullTrace[::self.factor]):
            self.ft.axes.plot(downsampled_onset,
                              0,
                              marker="|",
                              markersize=500,
                              markeredgecolor='g')

        self.ft.draw()

        user_info = 'Anomaly number ' + str(self.currentOpenedAno + 1)
        self.update_label(1, user_info)

    def debug_display(self, index):
        print("affichage ano num {}".format(index))
        print("Onset", self.allCandidates[index].onset)
        print("Size seizure",
              self.allCandidates[index].end - self.allCandidates[index].onset)
        print("End seizure", self.allCandidates[index].end)

    def span_add(self, xmin, xmax):
        self.span = mwidgets.SpanSelector(self.sc.axes,
                                          self.onselect,
                                          "horizontal",
                                          useblit=True,
                                          props=dict(alpha=0.1,
                                                     facecolor="red"),
                                          interactive=True,
                                          drag_from_anywhere=True)
        self.span.extents = (xmin, xmax)
        self.span.onselect(xmin, xmax)

    def ft_update_display(self):
        # To check the gain in memory of the signal sampling in the timeline (commented part)
        # original_size = self.fullTrace.nbytes
        # print(f'Taille du signal original: {original_size} octets')

        downsampled_trace = self.fullTrace[::self.factor]

        # downsampled_size = downsampled_trace.nbytes
        # print(f'Downsampled signal size: {downsampled_size} octets')

        # # Memory gain
        # memory_gain = (original_size - downsampled_size) / original_size * 100
        # print(f'Memory gain: {memory_gain:.2f}%')

        # to keep anomalies despite undersampling
        downsampled_onsets = []
        for candidate in self.allCandidates:
            onset = candidate.onset // self.factor
            if onset < len(downsampled_trace):
                downsampled_onsets.append(onset)
        ####

        self.ft.axes.clear()
        self.ft.axes.plot(downsampled_trace,
                          marker='x',
                          markerfacecolor='r',
                          markeredgecolor='r',
                          markevery=downsampled_onsets,
                          markersize=10,
                          linewidth=0.5)

        self.ft.axes.add_patch(
            Rectangle((0, self.min_trace * 0.2),
                      len(downsampled_trace),
                      (abs(self.min_trace) + self.max_trace) * 0.2,
                      alpha=0.3,
                      edgecolor='orange',
                      facecolor='orange',
                      fill=True))

        self.ft.figure.canvas.draw()

    def display_step(self, step):
        if self.currentOpenedAno == -1:
            self.currentOpenedAno += 1
            self.displayAno(self.currentOpenedAno)
        elif self.currentOpenedAno == 0 and step == -1:
            self.displayAno(self.currentOpenedAno)
        elif self.currentOpenedAno == self.anoCount - 1:
            self.currentOpenedAno = 0
            self.displayAno(self.currentOpenedAno)
        else:
            self.currentOpenedAno += step
            self.displayAno(self.currentOpenedAno)

    def save_done(self):
        ###
        if not self.saveFolder or not path.isdir(self.saveFolder):
            print(
                "The save folder is not defined. The user needs to select the backup folder.")
            self.save_folder_button_clicked(None)
            if not self.saveFolder or not path.isdir(self.saveFolder):
                print("No valid save folder selected")
                return
        ###

        if path.isdir(self.saveFolder + "/seizures"):
            print("Save folder for seizures exists")
        else:
            os.mkdir(self.saveFolder + "/seizures")

        if path.isdir(self.saveFolder + "/anomalies"):
            print("Save folder for non-seizures exists")
        else:
            os.mkdir(self.saveFolder + "/anomalies")

        pouet = self.filePath.split("/")
        filename = pouet[len(pouet) - 1].split(".")
        poped_indices = []
        for i in np.array(self.selectedAno).nonzero()[0]:
            poped_indices.append(i)
            self.fullCSVPath = self.saveFolder + "/seizures" +\
                "/"+filename[0]+"_{}.csv".format(i+1)
            # PDF
            self.fullPDFPath = self.saveFolder + \
                "/seizures" + f"/{filename[0]}_{i+1}.pdf"

            # EDF
            self.fullEDFPath = f"{self.saveFolder}/seizures/{filename[0]}_{i+1}.edf"

            with open(self.fullCSVPath, 'ab') as f:
                np.savetxt(
                    f,
                    self.fullTrace[self.newMinMax[i][0]:self.newMinMax[i][1]],
                    delimiter=","
                )
            # PDF
            # reminder : save_as_pdf(self, pdf_path, data)
            self.save_pdf(
                self.fullPDFPath, self.fullTrace[self.newMinMax[i][0]:self.newMinMax[i][1]])

            # EDF
            self.save_edf(
                self.fullEDFPath, self.fullTrace[self.newMinMax[i][0]:self.newMinMax[i][1]])

            f.close()

        remaining = [x for x in range(self.anoCount) if x not in poped_indices]
        print("poped_indices ", poped_indices)
        print("remaining ano to save", remaining)
        for i in remaining:
            self.fullCSVPath = self.saveFolder + "/anomalies" + \
                "/"+filename[0]+"_{}.csv".format(i+1)
            # PDF
            self.fullPDFPath = self.saveFolder + \
                "/anomalies" + f"/{filename[0]}_{i+1}.pdf"
            # EDF
            self.fullEDFPath = f"{self.saveFolder}/anomalies/{filename[0]}_{i+1}.edf"

            with open(self.fullCSVPath, 'ab') as f:
                np.savetxt(
                    f,
                    self.fullTrace[self.allCandidates[i].onset:self.
                                   allCandidates[i].end],
                    delimiter=","
                )

            # reminder : save_as_pdf(self, pdf_path, data)
            self.save_pdf(
                self.fullPDFPath, self.fullTrace[self.allCandidates[i].onset:self.allCandidates[i].end])

            # EDF
            self.save_edf(
                self.fullEDFPath, self.fullTrace[self.allCandidates[i].onset:self.allCandidates[i].end])

            f.close()

        self.save_recap(filename)

        if self.background:
            self.save_background()
        else:
            print("no background")

    def check_uncheck(self):
        if self.selectedAno[self.currentOpenedAno] == 0:
            self.selectedAno[self.currentOpenedAno] = 1
            self.nbOfSelected += 1
        else:
            self.selectedAno[self.currentOpenedAno] = 0
            self.nbOfSelected -= 1

    def background_check(self):
        if self.background is False:
            self.background = True
            print("background:", self.background)
        else:
            self.background = False
            print("background:", self.background)

    def amp_seiz(self, ano):
        return max(self.fullTrace[self.newMinMax[ano][0]:
                                  self.newMinMax[ano][1]]) - \
            min(self.fullTrace[self.newMinMax[ano][0]: self.newMinMax[ano][1]])

    def on_mouse_move_display(self, event):
        if event.inaxes == self.ft.axes and self.traceProcessingDone and \
                event.ydata > self.min_trace*0.2 and event.ydata < self.max_trace*0.2:
            # print(event.xdata)

            # compensate for the decrease in resolution
            x = event.xdata * self.factor

            self.sc.axes.xaxis.set_major_locator(
                mticker.MultipleLocator(1000))
            self.sc.axes.yaxis.set_major_locator(
                mticker.MultipleLocator(0.1))
            newxlone = x -\
                int(self.allvalues[7]) * self.samplerate
            newxltwo = x +\
                int(self.allvalues[7]) * self.samplerate
            self.sc.axes.set_xlim(newxlone, newxltwo)

            leftpeek = (0, (int(x) - 10000) %
                        len(self.fullTrace))[int(x) - 10000 > 0]
            rightpeek = (len(self.fullTrace),
                         (int(x) + 10000 % len(self.fullTrace)
                          ))[int(x) + 10000 < len(self.fullTrace)]
            newylone = max(self.fullTrace[leftpeek:rightpeek])
            newyltwo = min(self.fullTrace[leftpeek:rightpeek])
            amp = newylone - newyltwo
            factor = amp * 0.1
            self.sc.axes.set_ylim(newyltwo - factor, newylone + factor)

            a = self.sc.axes.get_xticks().tolist()
            self.sc.axes.xaxis.set_major_locator(mticker.FixedLocator(a))
            newticks = [str(float(k) / (self.samplerate / 1000))
                        for k in a]
            self.sc.axes.set_xticklabels(newticks)
            self.sc.axes.set_ylabel('mV')
            self.sc.axes.set_xlabel('time (ms)')

            plt.tight_layout()

            self.sc.draw()

    def on_mouse_pos_keypress(self, event):
        if event.inaxes == self.sc.axes and self.traceProcessingDone and event.dblclick:
            print(event)
            # compensate for the decrease in resolution
            x = event.xdata * self.factor

            onset = (0,
                     (int(x) - 1000) % len(self.fullTrace))[int(x) - 1000 > 0]
            end = (len(self.fullTrace),
                   (int(x) +
                   1000 % len(self.fullTrace)))[int(x) +
                                                1000 < len(self.fullTrace)]
            self.allCandidates.append(SignalCandidate(onset, end, "manual"))
            self.anoCount += 1
            self.newMinMax.append([onset, end])
            self.addedAno += 1
            self.selectedAno.append(0)
            self.currentOpenedAno = self.anoCount - 1
            self.all_buttons['QCheckBox'][0].setChecked(False)
            self.all_buttons['QCheckBox'][0].repaint()
            print("Does this anomaly exists?",
                  self.allCandidates[self.currentOpenedAno])
            self.span_add(onset, end)
            self.sc.draw()
            print(self.allCandidates)

    def save_recap(self, filename):
        self.fullCSVPath = self.saveFolder + "/seizures" +\
            "/"+filename[0]+"_recap.csv"
        # PDF
        self.fullPDFPath = self.saveFolder + \
            "/seizures" + f"/{filename[0]}_recap.pdf"

        columnName = [
            "file", "ID", "sample_rate", "onset_downsampled",
            "end_downsampled", "onset", "end", "amplitude", "duration",
            "cumulative duration", "frequency"
        ]

        f = open(self.fullCSVPath, 'w')
        writer = csv.writer(f)
        writer.writerow(columnName)
        cumul = 0
        freq = (self.dataLength / 600000000) * self.nbOfSelected
        for i in np.array(self.selectedAno).nonzero()[0]:
            row = []
            row.append(filename[0])
            row.append(i + 1)
            row.append(self.samplerate)
            row.append(self.newMinMax[i][0])
            row.append(self.newMinMax[i][1])
            row.append(self.newMinMax[i][0] * 5)
            row.append(self.newMinMax[i][1] * 5)
            row.append(self.amp_seiz(i))
            row.append((self.newMinMax[i][1] - self.newMinMax[i][0]) /
                       (self.samplerate / 1000))
            cumul += (self.newMinMax[i][1] -
                      self.newMinMax[i][0]) / (self.samplerate / 1000)
            row.append(cumul)
            row.append(freq)
            writer.writerow(row)
        #
        self.save_recap_pdf(self.fullPDFPath, columnName, filename)
        #
        f.close()

    def save_background(self):
        backgrnd = "/background"
        pouet = self.filePath.split("/")
        filename = pouet[-1].split(".")
        seiz = np.array(self.selectedAno).nonzero()[0]

        if path.isdir(self.saveFolder + backgrnd):
            print("Save folder for background exists")
        else:
            os.mkdir(self.saveFolder + backgrnd)

        self.fullCSVPath = self.saveFolder + \
            backgrnd + f"/{filename[0]}_background_1.csv"
        self.fullPDFPath = self.saveFolder + \
            backgrnd + f"/{filename[0]}_background_1.pdf"
        self.fullEDFPath = f"{self.saveFolder}{backgrnd}/{filename[0]}_background_1.edf"

        f = open(self.fullCSVPath, 'ab')
        np.savetxt(
            f,
            self.fullTrace[0:self.newMinMax[seiz[0]][0]],
            delimiter=","
        )

        self.save_pdf(self.fullPDFPath,
                      self.fullTrace[0:self.newMinMax[seiz[0]][0]])
        self.save_edf(self.fullEDFPath,
                      self.fullTrace[0:self.newMinMax[seiz[0]][0]])

        f.close()

        for i in range(1, len(seiz) - 1):

            self.fullCSVPath = self.saveFolder + backgrnd + \
                f"/{filename[0]}_background_{seiz[i]+1}.csv"
            self.fullPDFPath = self.saveFolder + backgrnd + \
                f"/{filename[0]}_background_{seiz[i]+1}.pdf"
            self.fullEDFPath = f"{self.saveFolder}{backgrnd}/{filename[0]}_background_{seiz[i]+1}.edf"

            if len(self.fullTrace[self.newMinMax[seiz[i]][1]:self.newMinMax[seiz[i + 1]][0]]) < 4096:
                f = open(self.fullCSVPath, 'ab')
                np.savetxt(
                    f,
                    self.fullTrace[self.newMinMax[seiz[i]][1]:self.
                                   newMinMax[seiz[i + 1]][0]],
                    delimiter=","
                )

                self.save_pdf(
                    self.fullPDFPath, self.fullTrace[self.newMinMax[seiz[i]][1]:self.newMinMax[seiz[i + 1]][0]])
                self.save_edf(
                    self.fullEDFPath, self.fullTrace[self.newMinMax[seiz[i]][1]:self.newMinMax[seiz[i + 1]][0]])

                f.close()

        self.fullCSVPath = self.saveFolder + backgrnd + \
            f"/{filename[0]}_background_{seiz[-1]}.csv"
        self.fullPDFPath = self.saveFolder + backgrnd + \
            f"/{filename[0]}_background_{seiz[-1]}.pdf"
        self.fullEDFPath = f"{self.saveFolder}{backgrnd}/{filename[0]}_background_{seiz[-1]}.edf"

        f = open(self.fullCSVPath, 'ab')
        np.savetxt(
            f,
            self.fullTrace[self.newMinMax[seiz[-1]][1]:len(self.fullTrace)],
            delimiter=","
        )

        self.save_pdf(
            self.fullPDFPath, self.fullTrace[self.newMinMax[seiz[-1]][1]:len(self.fullTrace)])
        self.save_edf(
            self.fullEDFPath, self.fullTrace[self.newMinMax[seiz[-1]][1]:len(self.fullTrace)])

        f.close()

    def save_pdf(self, pdf_path, data):
        # to generate PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        # to position text
        _, h = letter
        c.drawString(30, h - 40, f"Data for {pdf_path}")
        # browse the data and convert it to a string
        for i, row in enumerate(data):
            text = ','.join(map(str, row)) if isinstance(
                row, (list, np.ndarray)) else str(row)
            # add to pdf document with correct placement
            c.drawString(30, h - 60 - i * 10, text)
        c.save()

    def save_recap_pdf(self, pdf_path, columnName, filename):
        # for pdf
        c = canvas.Canvas(pdf_path, pagesize=letter)
        _, h = letter
        # 'title' line
        c.drawString(30, h - 40, f"Recap for {filename[0]}")
        # column names as strings
        text = ','.join(columnName)
        # adding string to pdf
        c.drawString(30, h - 60, text)
        c.save()

    def save_edf(self, edf_path, data):
        channels = 1
        rate = 256  # data sampling frequency
        # pyedflib module used to write data to edf file
        with pyedflib.EdfWriter(edf_path, channels, file_type=pyedflib.FILETYPE_EDF) as edf_writer:
            min = float(f"{np.min(data):.5f}")
            max = float(f"{np.max(data):.5f}")

            channel_info = {
                'label': 'channel_1',
                'dimension': 'uV',
                'sample_rate': rate,
                'physical_min': min,
                'physical_max': max,
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            }
            # def channel info
            edf_writer.setSignalHeader(0, channel_info)
            # save data to EDF file
            edf_writer.writeSamples([data])
            # if the open file is EDF, we retrieve the data from its header to put them in the new
            if self.filePath.lower().endswith('.edf'):
                header = {
                    'patientname': self.edf.getPatientName(),
                    'recording_additional': self.edf.getRecordingAdditional(),
                    'startdate': self.edf.getStartdatetime(),
                    'admincode': self.edf.getAdmincode(),
                    'technician': self.edf.getTechnician(),
                    'equipment': self.edf.getEquipment(),
                    'patientcode': self.edf.getPatientCode(),
                    'birthdate': self.edf.getBirthdate(),
                    'gender': self.edf.getSex(),
                    'sex': self.edf.getSex(),
                    'startdate_night': '',
                    'patient_additional': self.edf.getPatientAdditional(),
                }
            # if the open file is ABF, we retrieve the metadata to put them in the header of the new EDF file
            elif self.filePath.lower().endswith('.abf'):
                abf_version = self.abf.abfVersion
                abf_file_path = self.abf.abfFilePath
                abf_id = self.abf.abfID

                # header components are limited to 80 characters so we reduce the data entered in the header to 80 characters
                def limit_string(s):
                    return s if len(s) <= 80 else s[:80]
                recording_additional = limit_string(
                    f'ABF ID: {abf_id}, Ver: {abf_version}')
                equipment = limit_string(f'Path: {abf_file_path}')

                header = {
                    'patientname': 'zebra',
                    'recording_additional': recording_additional,
                    'startdate': self.abf.abfDateTime,
                    'admincode': '',
                    'technician': '',
                    'equipment': equipment,
                    'recording_id': '',
                    'patientcode': '',
                    'birthdate': '',
                    'gender': '',
                    'sex': '',
                    'startdate_night': '',
                    'additional': str(self.abf.dataRate),
                    'patient_additional': '',

                }

            edf_writer.setHeader(header)
            edf_writer.close()

    def on_dblclick_select_timeline(self, event):
        if event.inaxes == self.ft.axes and self.traceProcessingDone and event.dblclick:

            # Converts click position to scale following the decrease in resolution
            # so that the right click always takes you to the anomaly to the right of the cursor
            convert_click = int(event.xdata * self.factor)

            sliced = [i.onset for i in self.allCandidates]
            index_select = bisect.bisect_left(sliced, convert_click)

            self.displayAno(index_select)

    def closeEvent(self, event):
        x = datetime.datetime.now()
        if self.savedconfig is False:
            filename = self.filePath.split('/')[-1][:-4]
            self.save_config("param_" + filename + x.strftime("%Y%m%d%H") +
                             ".config~")
        event.accept()
