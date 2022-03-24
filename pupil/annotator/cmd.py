from functools import partial
from typing import Any, List

from IPython.display import display
from ipywidgets import Button, Dropdown, HBox, Output
from pupil.db.database import DataBase
from pupil.sampler import Sampler


class CMD:
    def __init__(self, db: DataBase, sampler: Sampler):
        """_summary_

        Args:
            db (DataBase): _description_
            sampler (Sampler): _description_
        """
        self.db = db
        self.sampler = sampler

    def do_it(self):
        """_summary_"""
        ind = self.sampler.sample(n=5)
        print("Index: ", ind, self.db.metadb.label, self.db.get(ind, return_emb=False))
        inp = int(input("Label?"))
        self.db.metadb.set_label(ind, inp)


class Annotator:
    def __init__(self, labels: List[str], data_col_name: str, db=DataBase):
        self.labels = labels
        self.col_name = data_col_name
        self.db = db
        self.output = Output()
        self.buttons = self.labels_buttons(self.labels)

    def drop_down(self, options: List[str]) -> List[Button]:
        self.dd = Dropdown(options=options)
        btn = Button(description="submit")
        btn.on_click(partial(self.callback, value=self.dd))
        return [self.dd, btn]

    def toggle_button(self, options: List[str]) -> List[Button]:
        buttons = []
        for label in options:
            btn = Button(description=label)
            btn.on_click(partial(self.callback, value=label))
            buttons.append(btn)
        return buttons

    def labels_buttons(self, options: List[str], skip: bool = True) -> HBox:
        use_dropdown = len(options) > 5
        buttons = []
        if use_dropdown:
            buttons = self.drop_down(options)
        else:
            buttons = self.toggle_button(options)
        if skip:
            skip_btn = Button(description="Skip")
            skip_btn.on_click(lambda x: self.show_next())
            buttons.append(skip_btn)

        box = HBox(buttons)
        return box

    def callback(self, b, value: Any) -> None:
        if hasattr(value, "value"):
            value = value.value
        self.db.metadb.set_label(self.ind, value)  # type: ignore
        self.show_next()

    def disp(self, ind: int) -> None:
        self.output.clear_output()
        data = self.db.metadb[ind][0].__dict__[self.col_name]  # type: ignore
        self.output.append_stdout(data)
        display(self.output)

    def show_next(self) -> None:
        if self.n < self.max_n:
            self.ind = self.inds[self.n]
            self.disp(self.ind)
            self.n += 1
        else:
            self.output.clear_output()
            self.output.append_stdout("No more data to show")

    def annotate(self, inds: List[int]):
        self.ind = None
        self.n = 0
        self.inds = inds
        self.max_n = len(inds)
        display(self.buttons)
        self.show_next()
