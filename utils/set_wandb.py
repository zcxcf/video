import wandb


def set_wandb(args, name):
    wandb.login(key='73283fc4b55614adc5934a0812ba6622a1aa5301')

    wandb.init(
        project=name,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        name=args.model+args.dataset+'_lr'+str(args.lr)+str(args.min_lr)+'_epoch'+str(args.epochs)
    )

