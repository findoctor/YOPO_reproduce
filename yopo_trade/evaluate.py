import torch
import config
import train
import attack


def eval(model, test_loader, attack_flag = True):
    # Set model to eval phase
    model.eval()

    clean_acc = 0
    final_acc = 0
    
    # For each batch, calculate the accuracy
    for step, (x, label) in enumerate(test_loader):
        x = x.to(config.Parameter_setting['device'])
        label = label.to(config.Parameter_setting['device'])

        # Forward Pass
        y_pred = model(x)

        # Calculate the accuracy
        acc = train.torch_accuracy(y_pred, label, (1,))
        
        # Update total accuracy
        clean_acc =  (clean_acc * step + acc[0].item()) / (step + 1)

        # If using attack method, calculate the final_acc
        if attack_flag:
            final_x = attack.attacking(x, label, model)

            with torch.no_grad():
                y_pred = model(final_x)
                acc = train.torch_accuracy(y_pred, label, (1,))
                final_acc = (final_acc * step + acc[0].item()) / (step + 1)

        if step % config.Parameter_setting['eval_print_step'] == 0:
                print("Batch step "+str(step)+" Clean_acc: "+str(clean_acc)+" Final_acc: "+str(final_acc))

    return clean_acc, final_acc