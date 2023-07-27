from IPython.display import display_html
display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
    
num_of_runs = 5
for i in range(1,num_of_runs+1):
    
    import warnings
    warnings.simplefilter("ignore")

    import uNetMain

    print('run',i)

    in_channel = 1 
    first_out_channel = 32
    trn_folder = '../../data/bucket2/tr'
    val_folder = '../../data/bucket2/val'
    goldBinary_folder = '../../data/goldsBinaryAll'
    lr=1.0
    patience = 40
    min_delta = 0.0

    model_name_pre = 'weights/allWeightsB2'
    initial_model_pre = '../base/25_weights/allWeightsB2'
    alpha_s = 0.000005; beta_s = 0.0; alpha_m = 0.000005; beta_m = 0.0; alpha_f = 0.0; beta_f = 0.0001
    
    if i==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(i) + '.pth'
    
    if i==1:
        initial_model = initial_model_pre + '.pth'
    else:
        initial_model = initial_model_pre + '_' + str(i) + '.pth'

    uNetMain.callMain(in_channel, first_out_channel, trn_folder, val_folder, goldBinary_folder, lr, patience, min_delta, model_name, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f, initial_model)

    from IPython.display import display_html
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

import IoU_Pixlws_WeighHausdrffByType as evl
goldBinary_folder = '../../data/goldsBinaryAll'
ts_folder = '../../data/bucket2/ts'
model_name_pre = 'weights/allWeightsB2'
num_of_runs = 5
threshold = 0.5
excel_name = 'excels/'+model_name_pre.split('/')[1]+'.xlsx'
print('creating allWeightsB2 scores')
evl.create_result_excel(num_of_runs, threshold, goldBinary_folder, ts_folder, model_name_pre, excel_name)
