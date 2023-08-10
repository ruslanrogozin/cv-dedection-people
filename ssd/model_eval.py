import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def model_evaluate(model, encoder, val_dataloader, device):
    target = []
    preds = []
    for _, data in enumerate(tqdm(val_dataloader,
                                  total=len(val_dataloader))):
        model.eval()
        img, _, images_sizes, bbox_data, bbox_labels = data
        with torch.no_grad():
            ploc, plabel = model(img.to(device))
            detections = encoder.decode_batch(ploc, plabel, 0.5, 200)

        for idx in range(ploc.shape[0]):
            true_dict = dict()
            preds_dict = dict()
            htot, wtot = images_sizes[0][idx].item(
            ), images_sizes[1][idx].item()
            pred_bbx = detections[idx][0]
            tr_bbx = bbox_labels[idx] > 0
            bbx_target = []
            for j in range(len(bbox_data[idx][tr_bbx])):
                l, t, r, b = bbox_data[idx][tr_bbx][j].detach().cpu().numpy()
                bbx_target.append([l * wtot, t * htot, r * wtot, b * htot])

            bbx_pred = []
            for j in range(pred_bbx.shape[0]):
                l, t, r, b = pred_bbx[j].detach().cpu().numpy()
                bbx_pred.append([l * wtot, t * htot, r * wtot, b * htot])

            true_dict["boxes"] = torch.tensor(bbx_target).detach().cpu()
            true_dict["labels"] = bbox_labels[idx][tr_bbx].detach().cpu()
            preds_dict["boxes"] = torch.tensor(bbx_pred).detach().cpu()
            preds_dict["scores"] = detections[idx][2].detach().cpu()
            preds_dict["labels"] = detections[idx][1].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        # if nbatch == 4:
        # break

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    return metric.compute()
