import train as trn
import matplotlib.pyplot as plt

def callMain(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta, model_name,data_type):
   
    losses, val_losses = trn.train(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta,model_name,data_type)

    fig, ax = plt.subplots()
    plt.xlabel('epochs')
    plt.ylabel('losses')
    ax.plot(losses, label='trn')
    ax.plot(val_losses, label='val')
    legend = ax.legend(shadow=True, fontsize='x-large')
    plt.show()
    graph_name = model_name.split('/')
    graph_name = 'graphs/'+graph_name[len(graph_name)-1].split('.')[0]+'.png'
    plt.savefig(graph_name)