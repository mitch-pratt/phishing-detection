from src.interface.cli import main_menu
from src.pipeline.data_utils import initialise_models
from src.session.session import Session

from src.pipeline.pipeline import (
    ensure_dataset,
    run_classifier_mode,
    run_experiment_mode,
)

session = Session()
ensure_dataset(session)
initialise_models(session)


while True:
    choice = main_menu()

    if choice == "1":
        run_experiment_mode(session)

    elif choice == "2":
        run_classifier_mode(session)

    elif choice == "0":
        print("Exiting...")
        break

