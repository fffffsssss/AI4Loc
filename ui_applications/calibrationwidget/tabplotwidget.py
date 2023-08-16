# Tab plot widget
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use('qt5agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class TabPlotWidget(QWidget):
    """
    Tab plot widget
    """

    def __init__(self, parent=None):
        """
        Constructor.
        :param parent: Parent widget
        """
        QWidget.__init__(self, parent)
        self.tab_widget = QTabWidget(self)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.tab_widget)
        self.setLayout(self.layout)

        self.toolbar_list = []
        self.canvas_list = []
        self.figure_list = []

    def add_plot(self, title, figure):
        """
        Add plot.
        :param title: Title.
        :param figure: Figure.
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        figure.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.91, wspace=0.2, hspace=0.2)
        # figure.constrained_layout = True
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, widget)

        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        self.tab_widget.addTab(widget, title)

        self.toolbar_list.append(toolbar)
        self.canvas_list.append(canvas)
        self.figure_list.append(figure)

    def clear(self):
        """
        Clear all figures.
        """
        self.tab_widget.clear()
        self.toolbar_list.clear()
        self.canvas_list.clear()
        self.figure_list.clear()
