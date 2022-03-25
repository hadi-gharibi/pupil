from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import partial
from typing import Any, List

from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, HBox, Image, Output, VBox
from numpy import disp

from pupil.db.database import DataBase

draw_line = (
    lambda s: f"""<html><body>
    <hr align="left" style="width:{s}%;height:1px;border-width:0;color:gray;background-color:gray">
</body></html>
"""
)


show_txt_data = (
    lambda txt: f"""
    <h1 style="color:grey;font-size:130%;">Data:</h1>
    
    <b style="color:black;">{txt}</b>
"""
)


def show_img_data(path):
    file = open(path, "rb")
    image = file.read()
    return image


class DataType(Enum):
    TEXT = auto()
    IMAGE = auto()


class Annotator(ABC):
    def __init__(
        self,
        labels: List[str],
        data_type: DataType = DataType.TEXT,
    ) -> None:

        self.labels = labels
        # self.col_name = data_col_name
        # self.db = db
        self._output = Output()
        # LabelsBox
        self._buttons = self._labels_buttons(self.labels)
        self._labelbox = VBox(
            [
                HTML(value=draw_line(40)),
                HTML(value="Labels: "),
                HBox(self._buttons),
                HTML(value=draw_line(40)),
            ],  # type: ignore
            background_color="grey",
        )

        # ControlBox
        self._back_btn = self._set_btn(
            callback=self._back_callback, description="Back", disabled=True
        )
        self._skip_btn = self._set_btn(callback=self._skip_callback, description="Skip")
        self._del_btn = self._set_btn(
            callback=self._del_callback, description="Delete", button_style="danger"
        )
        self._controlbox = VBox(
            [
                HBox(
                    [
                        self._back_btn,
                        self._skip_btn,
                        self._del_btn,
                    ]  # type: ignore
                ),
                HTML(value=draw_line(40)),
            ]
        )

        # current label box
        self._current_label = HTML(value="")

        self._current_labelbox = VBox([self._current_label, HTML(value=draw_line(15))])  # type: ignore

        # ShowDataBox
        if data_type == DataType.TEXT:
            self._show_data = HTML(value="")
            self._set_data = show_txt_data
        elif data_type == DataType.IMAGE:
            self._show_data = Image(value=b"")
            self._show_data.layout.object_fit = "scale-down"
            self._set_data = show_img_data
            self._show_data.layout.height = "400px"
            self._show_data.layout.weight = "400px"

        # box aggs
        self._actions_box = VBox([self._labelbox, self._controlbox])  # type: ignore
        self._data_box = VBox([self._current_labelbox, self._show_data])  # type: ignore

        self.labeld_data = {}

    def _drop_down(self, options: List[str]) -> List:
        self.dd = Dropdown(options=options)
        btn = Button(description="submit", button_style="primary")
        btn.on_click(partial(self._submit_callback, value=self.dd))
        return [self.dd, btn]

    def _toggle_button(self, options: List[str]) -> List:
        buttons = []
        for label in options:
            btn = Button(description=label, button_style="primary")
            btn.on_click(partial(self._submit_callback, value=label))
            buttons.append(btn)
        return buttons

    def _labels_buttons(self, options: List[str], skip: bool = True) -> HBox:
        use_dropdown = len(options) > 5
        buttons = []

        if use_dropdown:
            buttons = self._drop_down(options)
        else:
            buttons = self._toggle_button(options)
        return buttons  # type: ignore

    def _set_btn(self, callback, **kwargs):
        btn = Button(**kwargs)
        btn.on_click(callback)
        return btn

    def _del_callback(self, a):
        self.labeld_data.pop(self._ind, None)

    def _back_callback(self, a) -> None:
        self._skip_btn.disabled = False
        if self._n == 0:
            self._back_btn.disabled = True
            return
        self._n -= 1
        self._ind = self._inds[self._n]
        self._disp(self._ind)

    def _submit_callback(self, b, value: Any) -> None:
        if hasattr(value, "value"):
            value = value.value
        self.labeld_data[self._ind] = value
        self._back_btn.disabled = False
        self._show_next()

    def _skip_callback(self, a) -> None:
        self._back_btn.disabled = False
        self._show_next()

    def _disp(self, ind: int) -> None:
        data = self.get_data(ind)
        self._current_label.value = (
            f"Current label: <mark>{self.labeld_data.get(self._ind, None)}</mark>"
        )
        self._show_data.value = self._set_data(data)
        display(self._data_box)

    def _show_next(self) -> None:
        if self._n < self._max_n:
            self._n += 1
            self._ind = self._inds[self._n]
            self._disp(self._ind)
        else:
            self._skip_btn.disabled = True
            self._output.clear_output()
            self._output.append_stdout("No more data to show")

    def annotate(self, inds: List[int]):
        self._ind = None
        self._n = -1
        self._inds = inds
        self._max_n = len(inds)

        display(self._actions_box)
        self._show_next()

    @abstractmethod
    def get_data(self, i) -> str:
        pass


class DataFrameAnnotator(Annotator):
    def __init__(self, df, col_name, **kwargs):
        self.df = df
        self.col_name = col_name
        super().__init__(**kwargs)

    def get_data(self, i):
        return self.df.iloc[i][self.col_name]


class PupilDBAnnotator(Annotator):
    def __init__(self, db: DataBase, col_name: str, **kwargs):
        self.db = db
        self.col_name = col_name
        super().__init__(**kwargs)

    def get_data(self, i) -> str:
        return self.db.metadb[i][0].__dict__[self.col_name]
