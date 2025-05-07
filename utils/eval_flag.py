import torch
from tqdm import tqdm
import wandb
import torch.distributed as dist


def eval_mat(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(flag=flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val Acc_"+flag: accuracy})

            pbar.close()

def eval_mat_mlp_mha(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(flag=flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "mlp_Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "mlp_Val Acc_"+flag: accuracy})

            pbar.close()

def eval_mat_fined(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    dict = {3:'s', 6:'m', 12:'l'}
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork_fined(flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            flag = dict[flag]
            wandb.log({"Epoch": epoch + 1, "Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val Acc_"+flag: accuracy})

            pbar.close()


def eval_mat_combined(model, valDataLoader, criterion, epoch, optimizer, args, flag, **kwargs):

    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "val/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val/Val Acc_" + flag: accuracy})
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict

# ------------ helper (和前面示例一致) ---------------------------------- #
def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
# ----------------------------------------------------------------------- #


def eval_mat_combined_dis(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    args: Any,
    flag: str,
    device: torch.device,
    local_rank: int,
    sub_dim,
    depth_list,
    mlp_ratio,
    mha_head,
) -> None:

    is_main = local_rank == 0
    bare_model = unwrap_ddp(model)

    # 固定子网拓扑
    bare_model.configure_subnetwork(sub_dim=sub_dim,
                        depth_list=depth_list,
                        mlp_ratio=mlp_ratio,
                        mha_head=mha_head,)
    bare_model.eval()

    total_loss: float = 0.0
    correct: int = 0
    total: int = 0
    V = 3  # 每个样本的视图数（写成常量，易改）

    with torch.no_grad(), tqdm(
        total=len(val_loader), ncols=100, disable=not is_main
    ) as pbar:
        pbar.set_description(f"Eval {epoch+1}/{args.epochs}")

        for imgs, labels in val_loader:
            # imgs: [B, V, C, T, H, W]   labels: [B]
            B, V = labels.size(0), imgs.size(1)

            imgs = imgs.view(B * V, *imgs.shape[2:]).to(device)  # (B*V, C, T, H, W)
            labels = labels.to(device)

            logits = bare_model(imgs)  # (B*V, num_cls)
            logits = logits.view(B, V, -1).mean(1)  # (B,   num_cls)

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=optimizer.param_groups[0]["lr"])
                pbar.update()

    # --- 全局聚合 ---------------------------------------------------------------- #
    tensor_correct = torch.tensor(correct, dtype=torch.long, device=device)
    tensor_total   = torch.tensor(total,   dtype=torch.long, device=device)

    dist.all_reduce(tensor_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(tensor_total,   op=dist.ReduceOp.SUM)

    if is_main:
        global_acc = (tensor_correct / tensor_total).item()
        epoch_loss = total_loss / tensor_total.item()

        print(f"[Val] flag={flag}  Acc={global_acc*100:.2f}%  "
              f"Loss={epoch_loss:.4f}")

        import wandb  # 延迟导入，避免非主进程误用
        wandb.log({
            "epoch": epoch + 1,
            f"val_acc_{flag}": global_acc,
            "val_loss": epoch_loss,
        })

# def eval_mat_combined_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag, device, local_rank, **kwargs):
#
#     with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
#         pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')
#
#         model.eval()
#
#         with torch.no_grad():
#             total_loss = 0.0
#             correct = 0
#             total = 0
#             for batch_idx, (img, label) in enumerate(valDataLoader):
#                 img = img.to(device)
#                 label = label.to(device)
#
#                 model.module.configure_subnetwork(**kwargs)
#                 preds = model(img)
#
#                 V = 3
#                 B = label.size(0)
#
#                 preds = preds.view(V, B, -1).mean(0)  # [B, num_cls]
#
#                 loss = criterion(preds, label)
#                 total_loss += loss.item()
#
#                 _, predicted = torch.max(preds, 1)
#                 total += label.size(0)
#                 correct += (predicted == label).sum().item()
#
#                 pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
#                 pbar.update(1)
#
#             epoch_loss = total_loss / len(valDataLoader)
#
#             correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
#             total = torch.tensor(total, dtype=torch.float32, device='cuda')
#
#             dist.all_reduce(correct, op=dist.ReduceOp.SUM)
#             dist.all_reduce(total, op=dist.ReduceOp.SUM)
#
#             if local_rank==0:
#                 accuracy = correct / total
#                 print(f"Global Accuracy: {accuracy.item()}")
#
#                 wandb.log({"Epoch": epoch + 1, "Val Acc_" + str(local_rank) + flag: accuracy.item()})
#                 wandb.log({"Epoch": epoch + 1, "Val epoch Loss": epoch_loss})


def eval_mat_dynamic(model, valDataLoader, criterion, epoch, optimizer, args, flag, sub_dim, depth_list, mlp_ratio,
                                           mha_head, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head, latency=latency)

                preds, mask = model(img)
                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency - mask)
                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})
def eval_mask(model, valDataLoader, criterion, epoch, optimizer, args, flag, sub_dim, depth_list, mlp_ratio,
                                           mha_head, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head, latency=latency)

                preds, attn_mask, mlp_mask = model(img)
                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-attn_mask) + torch.square(latency-mlp_mask)

                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
def eval_all(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_latency(latency=latency)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency - attn_mask) + torch.square(latency - mlp_mask) + torch.square(
                    latency - embed_mask) + torch.square(latency - depth_mlp_mask) + torch.square(
                    latency - depth_attn_mask)

                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})
def eval_macs_loss(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                # if epoch < 75 :
                #     t = 5
                # else:
                #     t = 1
                t = 1 + epoch / 200 * 4
                model.configure_latency(latency=latency, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-total_macs)

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)

            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})

            wandb.log({"Epoch": epoch + 1, "Val macs_" + flag: val_macs})

            wandb.log({"Epoch": epoch + 1, "Val acc/macs rate_" + flag: accuracy/val_macs})

def eval_macs_loss_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency, device, local_rank):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                t = 1 + epoch / 200 * 4

                model.module.configure_latency(latency=latency, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-total_macs)

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            if local_rank==0:
                accuracy = correct / total
                print(f"Global Accuracy: {accuracy.item()}")

                wandb.log({"Epoch": epoch + 1, "Val Acc_" + str(local_rank) + flag: accuracy.item()})

                wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
                wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
                wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

                wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

                wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
                wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
                wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

                wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
                wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})

                wandb.log({"Epoch": epoch + 1, "Val macs_" + flag: val_macs})

                wandb.log({"Epoch": epoch + 1, "Val acc/macs rate_" + flag: accuracy/val_macs})


def eval_macs_loss_dis_early_constrant(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency, device, local_rank):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                t = 1 + epoch / 200 * 4

                model.module.configure_latency(latency=latency, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                attn_mask = torch.mean(attn_mask)
                mlp_mask = torch.mean(mlp_mask)
                embed_mask = torch.mean(embed_mask)
                depth_mlp_mask = torch.mean(depth_mlp_mask)
                depth_attn_mask = torch.mean(depth_attn_mask)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-total_macs)

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            if local_rank==0:
                accuracy = correct / total
                print(f"Global Accuracy: {accuracy.item()}")
                wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
                wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
                wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

                wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

                wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
                wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
                wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

                wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
                wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})

                wandb.log({"Epoch": epoch + 1, "Val macs_" + flag: val_macs})

                wandb.log({"Epoch": epoch + 1, "Val acc/macs rate_" + flag: accuracy/val_macs})

def eval_router(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                t = 1

                model.configure_latency(latency=latency, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                attn_mask = torch.mean(attn_mask)
                mlp_mask = torch.mean(mlp_mask)
                embed_mask = torch.mean(embed_mask)
                depth_mlp_mask = torch.mean(depth_mlp_mask)
                depth_attn_mask = torch.mean(depth_attn_mask)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-total_macs)

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            accuracy = correct / total
            print(f"Global Accuracy: {accuracy.item()}")
            wandb.log({"Epoch": epoch + 1, "val_loss/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "acc/Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "val_attn_mask/Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "val_mlp_mask/Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_embed_mask/Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "val_depth_mlp_mask/Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_depth_attn_mask/Val depth attn mask_" + flag: val_depth_attn_mask})

            wandb.log({"Epoch": epoch + 1, "val_mac/Val macs_" + flag: val_macs})

            wandb.log({"Epoch": epoch + 1, "val_rate/macs rate_" + flag: accuracy/val_macs})



