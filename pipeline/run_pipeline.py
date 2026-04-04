from pipeline.train_models import run as train
from pipeline.generate_forecast import run as forecast
from pipeline.run_analysis import run as run_analysis

if __name__ == "__main__":
    run_analysis()
    train()
    forecast()