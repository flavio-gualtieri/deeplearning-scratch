# entrypoint.py

from deepscratch.config import DeepScratchConfig
from deepscratch.runs.context import RunContext

config = DeepScratchConfig.from_yaml("config.yaml")

run = RunContext(
    config=config,
    config_path="config.yaml",
)

run.setup()

data = build_data_module(config)
data.setup()

encoder = build_encoder(config, data=data)
head = build_head(config, data=data)
trainer = build_trainer(config)

pipeline = Pipeline(
    encoder=encoder,
    head=head,
    trainer=trainer,
)

pipeline.fit(
    train_loader=data.train_dataloader(),
    val_loader=data.val_dataloader(),
    run_context=run,
)