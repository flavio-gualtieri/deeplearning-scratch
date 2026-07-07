if run_context is not None:
    final_model_path = save_final_model(
        run_context=run_context,
        model=model,
        epoch=self.epochs,
        metrics=last_metrics,
        optimizer=optimizer,
        history=history,
    )

    model_summary_path = save_model_summary(
        run_context=run_context,
        model=model,
    )

    if logger is not None:
        logger.log_event(
            "final_model_saved",
            {
                "path": str(final_model_path),
                "epoch": self.epochs,
            },
        )

        logger.log_event(
            "model_summary_saved",
            {
                "path": str(model_summary_path),
            },
        )