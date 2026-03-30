from wm import App

from birdclef_2026_take_2.experiments.exp_001.training_job import Exp001

app = App.from_pyproject()
app.register(Exp001)
cli = app.cli
