import time

from config.config import Configs
from ssd.model_eval import model_evaluate
from ssd.train_one_loop import train_one_loop
from utils.utils import save_model


def train_model(
    model,
    encoder,
    num_epoch,
    optimizer,
    loss_func,
    train_dataloader,
    val_dataloader,
    device,
    scheduler=None,
    path_to_save=Configs.path_to_save_state_model,
):
    loss_mas = []
    map50 = []
    map_mas = []
    bes_map = 0
    for epoch in range(num_epoch):
        print(f"\nEPOCH {epoch+1} of {num_epoch}")
        start = time.time()
        losses = train_one_loop(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_dataloader=train_dataloader,
            device=device,
        )
        metrics_map = model_evaluate(
            model=model, encoder=encoder, val_dataloader=val_dataloader, device=device
        )

        cur_map = metrics_map["map"]

        if cur_map > bes_map:
            bes_map = cur_map
            save_model(
                model=model,
                optimizer=None,
                model_name="best_model_at_" + str(epoch),
                path=path_to_save,
                lr_scheduler=None,
            )

        print(f"Epoch #{epoch+1} train loss: {losses:.3f}")
        print(f"Epoch #{epoch+1} mAP: {metrics_map['map']}")
        print(f"Epoch #{epoch+1} mAP_50: {metrics_map['map_50']}")
        print(f"Epoch #{epoch+1} beset mAP: {bes_map}")

        loss_mas.append(losses)
        map50.append(metrics_map["map_50"])
        map_mas.append(metrics_map["map"])
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        if scheduler is not None:
            scheduler.step()

    save_model(
        model=model,
        optimizer=optimizer,
        model_name="end_state",
        path=path_to_save,
        lr_scheduler=scheduler,
    )

    with open("train_results.txt", "w") as file_handler:
        file_handler.write("loss\n")
        for item in loss_mas:
            file_handler.write("{}\t".format(item))

        file_handler.write("\nmap\n")
        for item in map50:
            file_handler.write("{}\t".format(item))

        file_handler.write("\nmap_50\n")
        for item in map_mas:
            file_handler.write("{}\t".format(item))
