import train as trn
import matplotlib.pyplot as plt

def callMain(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta, model_name, data_type,
            alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f, initial_model):
   
    losses, val_losses = trn.train(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta,model_name,data_type,alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f, initial_model)

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

def main(argv):
    
    in_channel = int(argv[0])
    first_out_channel = int(argv[1])
    trn_folder = argv[2]
    val_folder = argv[3]
    gold_folder = argv[4]
    lr = float(argv[5])
    patience = int(argv[6])
    min_delta = float(argv[7])
    model_name = argv[8]
    alpha_s =  argv[9]
    beta_s = argv[10] 
    alpha_o = argv[11] 
    beta_o = argv[12]

    callMain(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta, model_name,
            alpha_s, beta_s, alpha_o, beta_o)

if __name__ == "__main__":
    main(sys.argv[1:]) 
    
#!python uNetMain.py '3' '32' '../dataset/tr' '../dataset/val' '../dataset/golds' '0.01' '20' '0.0' 'model_weights.pth'    