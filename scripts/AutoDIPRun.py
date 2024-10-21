import os
import pathlib
import sys
dir_notebook = os.path.dirname(os.path.abspath("__file__"))
dir_parent = os.path.dirname(dir_notebook)
if not dir_parent in sys.path:
    sys.path.append(dir_parent)
from deepinterpolation.inference_collection import core_inference
from deepinterpolation.generator_collection import SingleTifGenerator
import tkinter.filedialog 
import numpy as np
import tifffile
import h5py
import tqdm
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QPainterPath, QBrush
from PyQt5.QtCore import Qt, QTimer, QItemSelection

import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(f"{parent_dir}optic")

from optic.config import *
from optic.manager import *
from optic.gui import *
from optic.io import *
from optic.utils import *
from optic.gui.bind_func import *

class AutoDIPRun(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.widget_manager = initManagers(WidgetManager())
        self.pre_post_frame = 30

        self.setupUI()

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.setGeometry(100, 100, 1200, 600)
        self.layout_main = QGridLayout(self.central_widget)
        self.layout_main.addLayout(self.makeLayoutMain(), 0, 0)

        self.bindFuncAllWidget()

    """
    makeLayout Function; Component
    小要素のLayout
    return -> Layout
    """
    def makeLayoutComponentButtons(self):
        layout = QHBoxLayout()
        layout.addWidget(self.widget_manager.makeWidgetButton(key="browse", label="Browse"))
        layout.addWidget(self.widget_manager.makeWidgetButton(key="remove", label="Remove"))
        layout.addWidget(self.widget_manager.makeWidgetButton(key="clear", label="Clear"))
        layout.addWidget(self.widget_manager.makeWidgetButton(key="run", label="Run"))
        layout.addWidget(self.widget_manager.makeWidgetButton(key="exit", label="Exit"))
        return layout
    
    def makeLayoutComponentListWidget(self, key, label, editable=False):
        layout = QVBoxLayout()
        layout.addWidget(self.widget_manager.makeWidgetLabel(key=key, label=label))
        layout.addWidget(self.widget_manager.makeWidgetListWidget(key=key, editable=editable))
        return layout

    """
    makeLayout Function; Section
    領域レベルの大Layout
    """
    def makeLayoutTop(self):
        layout = QHBoxLayout()
        layout.addLayout(self.makeLayoutComponentListWidget(key="input", label="Input TIF Files"))
        layout.addLayout(self.makeLayoutComponentListWidget(key="output", label="Output TIF Files", editable=True))
        return layout

    def makeLayoutBottom(self):
        layout = QHBoxLayout()
        layout.addLayout(self.makeLayoutComponentButtons())
        return layout
    
    def makeLayoutMain(self):
        layout = QVBoxLayout()
        layout.addLayout(self.makeLayoutTop())
        layout.addLayout(self.makeLayoutBottom())
        return layout
    
    """
    Functions
    """
    def dataIO(self):
        list_path_tif_input = self.browseTIFDirectory()
        for path_tif_input in list_path_tif_input:
            path_tif_output = self.convertFilePathInputToOutput(path_tif_input)
            self.addPathToListWidget(self.widget_manager.dict_listwidget["input"], path_tif_input)
            self.addPathToListWidget(self.widget_manager.dict_listwidget["output"], path_tif_output)

    def browseTIFDirectory(self):
        dir_path = openFolderDialog(self)
        list_path_tif = getMatchedPaths(getAllSubFiles(dir_path), list_str_include=[r"\.tif$"])
        return list_path_tif
    
    def convertFilePathInputToOutput(self, path_tif_input):
        path_tif_output = path_tif_input.rsplit(".", 1)[0] + "_denoised.tif"
        return path_tif_output
    
    def getParentDirectoryOfTIFFile(self, path_tif):
        return os.path.dirname(path_tif).replace("\\", "/")
    
    def addPathToListWidget(self, q_listwidget, path):
        q_listwidget.addItem(path)

    def getPathsFromListWidget(self, q_listwidget):
        return [q_listwidget.item(i).text() for i in range(q_listwidget.count())]

    def removePathFromListWidget(self, q_listwidget_input, q_listwidget_output, index):
        q_listwidget_input.takeItem(index)
        q_listwidget_output.takeItem(index)

    def clearListWidget(self):
        self.widget_manager.dict_listwidget["input"].clear()
        self.widget_manager.dict_listwidget["output"].clear()

    def modifyFrameNumber(self, file_path, batch_size):
        with tifffile.TiffFile(file_path) as tif:
            total_frames = len(tif.pages)
            number_frames = (total_frames - self.pre_post_frame * 2) // batch_size * batch_size
            end_frame = self.pre_post_frame + number_frames - 1
            return total_frames, number_frames, end_frame
        
    def paramsSetting(self):
        # パラメータの設定
        generator_param = {}
        inference_param = {}

        # ジェネレータのパラメータ設定
        generator_param["pre_post_frame"] = self.pre_post_frame
        generator_param["pre_post_omission"] = 0
        generator_param["steps_per_epoch"] = -1
        generator_param["train_path"] = self.path_input
        generator_param["batch_size"] = 10
        generator_param["start_frame"] = 0
        self.total_frames, self.number_frames, self.end_frame = self.modifyFrameNumber(generator_param["train_path"], generator_param["batch_size"])
        generator_param["end_frame"] = self.end_frame
        generator_param["randomize"] = 0

        # 推論のパラメータ設定
        inference_param["model_path"] = "D:/deepinterpolation/examples/models/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"
        # inference_param["model_path"] = "D:/deepinterpolation/examples/unet_single_1024_mean_absolute_error_2024_10_16_14_42_2024_10_16_14_42/2024_10_16_14_42_unet_single_1024_mean_absolute_error_2024_10_16_14_42_model.h5"

        inference_param["output_file"] = self.path_output.replace(".tif", ".h5")
        self.path_output_h5 = inference_param["output_file"]

        # ジェネレータとモデルの初期化
        data_generator = SingleTifGenerator(generator_param)
        inference_model = core_inference(inference_param, data_generator)
        return data_generator, inference_model
    
    def convertH5toPaddingTIF(self):
        h5 = h5py.File(self.path_output_h5, "r")
        arr = h5["data"]
        # padding
        arr_start = arr[0]
        arr_end = arr[-1]

        arr_stack_start = np.stack([arr_start] * self.pre_post_frame, axis=0)
        arr_stack_end = np.stack([arr_end] * (self.total_frames-self.end_frame-1), axis=0)

        arr = np.concatenate([arr_stack_start, arr, arr_stack_end], axis=0)
        h5.close()
        return arr

    def runDIP(self):
        list_path_input = self.getPathsFromListWidget(self.widget_manager.dict_listwidget["input"])
        list_path_output = self.getPathsFromListWidget(self.widget_manager.dict_listwidget["output"])
        for path_input, path_output in zip(list_path_input, list_path_output):
            print("Input:", path_input)
            print("Output:", path_output)
            self.path_input = path_input
            self.path_output = path_output
            dir_output = self.getParentDirectoryOfTIFFile(path_output)
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)

            data_generator, inference_model = self.paramsSetting()
            inference_model.run()
            arr = self.convertH5toPaddingTIF()
            tifffile.imwrite(path_output, arr)
            print("Image saved to", path_output)

        self.clearListWidget()
        print("DeepInterpolation completed.")
            


    """
    bindFunc Function
    配置したwidgetに関数を紐づけ
    """
    def bindFuncAllWidget(self):
        self.widget_manager.dict_button["browse"].clicked.connect(self.dataIO)
        self.widget_manager.dict_button["remove"].clicked.connect(lambda: self.removePathFromListWidget(
            self.widget_manager.dict_listwidget["input"], 
            self.widget_manager.dict_listwidget["output"],
            self.widget_manager.dict_listwidget["input"].currentRow()))
        self.widget_manager.dict_button["clear"].clicked.connect(self.clearListWidget)
        self.widget_manager.dict_button["run"].clicked.connect(self.runDIP)
        bindFuncExit(q_window=self, q_button=self.widget_manager.dict_button["exit"])


if __name__ == "__main__":
    app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
    gui = AutoDIPRun()
    gui.show()
    sys.exit(app.exec_())