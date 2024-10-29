import os
import pathlib
import sys
dir_notebook = os.path.dirname(os.path.abspath("__file__"))
dir_parent = os.path.dirname(dir_notebook)
if not dir_parent in sys.path:
    sys.path.append(dir_parent)
os.chdir(dir_parent)
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
    
    def makeLayoutComponentLoadFiles(self):
        layout = QVBoxLayout()
        layout.addLayout(makeLayoutLoadFileWidget(
            self.widget_manager, 
            label="Deep Interpolation Model", 
            key_label="browse_model", 
            key_lineedit="model", 
            key_button="browse_model"
        ))
        return layout
    
    def makeLayoutComponentListWidget(self, key, label, editable=False):
        layout = QVBoxLayout()
        layout.addWidget(self.widget_manager.makeWidgetLabel(key=key, label=label))
        layout.addWidget(self.widget_manager.makeWidgetListWidget(key=key, editable=editable))
        return layout
    
    def makeLayoutComponentCheckboxes(self):
        layout = QVBoxLayout()
        layout.addWidget(self.widget_manager.makeWidgetCheckBox(key="delete_h5", label="Delete h5 file after image export", checked=True))
        return layout

    """
    makeLayout Function; Section
    領域レベルの大Layout
    """
    def makeLayoutTop(self):
        layout = QHBoxLayout()
        layout.addLayout(self.makeLayoutComponentListWidget(key="input", label="Input TIF Directories", editable=False))
        layout.addLayout(self.makeLayoutComponentListWidget(key="output", label="Output TIF Directories", editable=True))
        return layout

    def makeLayoutBottom(self):
        layout = QVBoxLayout()
        layout.addLayout(self.makeLayoutComponentLoadFiles())
        layout.addLayout(self.makeLayoutComponentCheckboxes())
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
        dir_tif_input = self.browseTIFDirectory()
        dir_tif_output = self.convertFilePathInputToOutput(dir_tif_input)
        addItemToListWidget(self.widget_manager.dict_listwidget["input"], dir_tif_input, editable=False)
        addItemToListWidget(self.widget_manager.dict_listwidget["output"], dir_tif_output, editable=True)

    def browseTIFDirectory(self):
        dir_path = openFolderDialog(self)
        return dir_path
    
    def convertFilePathInputToOutput(self, dir_tif_input):
        dir_tif_output = dir_tif_input + "_denoised"
        return dir_tif_output

    def getPathsFromListWidget(self, q_listwidget):
        return [q_listwidget.item(i).text() for i in range(q_listwidget.count())]

    def removePathFromListWidget(self, q_listwidget_input, q_listwidget_output, index):
        q_listwidget_input.takeItem(index)
        q_listwidget_output.takeItem(index)

    def clearListWidget(self):
        clearListWidget(self.widget_manager.dict_listwidget["input"])
        clearListWidget(self.widget_manager.dict_listwidget["output"])

    # list_input_frames: list of input frames
    # total_frames: Total frames of the data
    # number_frames: Number of frames to be used for training
    # end_frame: End frame of the data
    def modifyFrameNumber(self, list_path_tif_input, batch_size):
        total_frames = 0
        list_input_frames = []
        for path_tif_input in list_path_tif_input:
            with tifffile.TiffFile(path_tif_input) as tif:
                total_frames += len(tif.pages)
                list_input_frames.append(len(tif.pages))
        number_frames = (total_frames - self.pre_post_frame * 2) // batch_size * batch_size
        start_frame = self.pre_post_frame
        end_frame = self.pre_post_frame + number_frames - 1
        return list_input_frames, total_frames, number_frames, start_frame, end_frame
        
    def paramsSetting(self):
        # パラメータの設定
        generator_param = {}
        inference_param = {}

        # 推論のパラメータ設定
        self.model_path = self.widget_manager.dict_lineedit["model"].text()
        inference_param["model_path"] = self.model_path

        # ジェネレータのパラメータ設定
        self.pre_post_frame = self.getPrePostFrame(inference_param["model_path"])
        generator_param["pre_post_frame"] = self.pre_post_frame
        generator_param["pre_post_omission"] = 0
        generator_param["steps_per_epoch"] = -1
        generator_param["train_path"] = self.dir_input
        generator_param["batch_size"] = 10
        generator_param["start_frame"] = 0
        self.list_input_frames, self.total_frames, self.number_frames, self.start_frame, self.end_frame = self.modifyFrameNumber(self.list_path_tif_input, generator_param["batch_size"])
        generator_param["end_frame"] = self.end_frame
        generator_param["randomize"] = 0

        inference_param["output_file"] = self.dir_output + ".h5"
        self.dir_output_h5 = inference_param["output_file"]

        # ジェネレータとモデルの初期化
        data_generator = SingleTifGenerator(generator_param)
        inference_model = core_inference(inference_param, data_generator)
        return data_generator, inference_model
    
    # get pre_post_frame from h5 file automatically
    def getPrePostFrame(self, path_h5py):
        with h5py.File(path_h5py, 'r') as f:
            dict_model_config = json.loads(f.attrs['model_config'].decode())
            pre_post_frame = dict_model_config["config"]["layers"][0]["config"]["batch_input_shape"][-1] // 2
        return pre_post_frame
    
    def convertH5toPaddingTIF(self):
        h5 = h5py.File(self.dir_output_h5, "r")
        arr = h5["data"]
        # padding
        arr_start = arr[0]
        arr_end = arr[-1]

        arr_stack_start = np.stack([arr_start] * self.pre_post_frame, axis=0)
        arr_stack_end = np.stack([arr_end] * (self.total_frames-self.end_frame-1), axis=0)

        arr = np.concatenate([arr_stack_start, arr, arr_stack_end], axis=0)
        h5.close()
        return arr
    
    def saveAsTIFList(self, arr):
        idx = 0
        for path_tif_output, len_tif in zip(self.list_path_tif_output, self.list_input_frames):
            arr_slice = arr[idx:idx+len_tif]
            tifffile.imwrite(path_tif_output, arr_slice)
            idx += len_tif
            print("Image saved to", path_tif_output)

    def runDIP(self):
        list_dir_input = self.getPathsFromListWidget(self.widget_manager.dict_listwidget["input"])
        list_dir_output = self.getPathsFromListWidget(self.widget_manager.dict_listwidget["output"])
        
        try:
            for dir_input, dir_output in zip(list_dir_input, list_dir_output):
                print("Input:", dir_input)
                print("Output:", dir_output)
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                self.dir_input = dir_input
                self.dir_output = dir_output
                self.list_path_tif_input = natsorted(getMatchedPaths(getAllSubFiles(dir_input), list_str_include=[r"\.tif$"]))
                self.list_path_tif_output = [path_tif_input.replace(self.dir_input, self.dir_output) for path_tif_input in self.list_path_tif_input]

                data_generator, inference_model = self.paramsSetting()
                print("Model:", self.model_path)
                inference_model.run()
                arr = self.convertH5toPaddingTIF()
                self.saveAsTIFList(arr)
                del arr
                if self.widget_manager.dict_checkbox["delete_h5"].isChecked():
                    os.remove(self.dir_output_h5)
                    print("h5 file deleted:", self.dir_output_h5)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

        self.clearListWidget()
        print("DeepInterpolation completed.")
            


    """
    bindFunc Function
    配置したwidgetに関数を紐づけ
    """
    def bindFuncAllWidget(self):
        bindFuncLoadFileWidget(
            q_widget=self, 
            q_button=self.widget_manager.dict_button["browse_model"], 
            q_lineedit=self.widget_manager.dict_lineedit["model"], 
            filetype=Extension.HDF5
        )
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