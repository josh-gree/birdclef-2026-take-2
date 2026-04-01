from wm import App

from birdclef_2026_take_2.experiments.exp_001.training_job import Exp001
from birdclef_2026_take_2.experiments.exp_002.training_job import Exp002
from birdclef_2026_take_2.experiments.exp_003.training_job import Exp003
from birdclef_2026_take_2.experiments.exp_004.training_job import Exp004
from birdclef_2026_take_2.experiments.exp_005.training_job import Exp005
from birdclef_2026_take_2.experiments.exp_006.training_job import Exp006

app = App.from_pyproject()
app.register(Exp001)
app.register(Exp002)
app.register(Exp003)
app.register(Exp004)
app.register(Exp005)
app.register(Exp006)
cli = app.cli
