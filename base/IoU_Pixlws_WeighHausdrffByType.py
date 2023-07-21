import helper_functions as hp
import numpy as np
import os
import cv2
import json
from skimage import metrics

def pixelwiseFPFNTP(pred,gold):
    TP = np.where(gold+pred==2,1,0).sum()
    FP = np.where(pred-gold==1,1,0).sum()
    FN = np.where(gold-pred==1,1,0).sum()

    return TP, FP, FN

def getPrecRecallFScore(TP, FP, FN):
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (TP + FN) > 0:    
        recall = TP / (TP + FN)
    else:
        recall = 0
    if (precision+recall) > 0:    
        f_score = (2*precision*recall)/(precision+recall)
    else:
        f_score = 0
        
    return precision, recall, f_score 

def IoU(prediction, goldB, threshold):
    goldB_cc = cv2.connectedComponents(goldB)
    pred_cc = cv2.connectedComponents(prediction)
    TP = 0
    for i in range(1,goldB_cc[0]+1):
        goldB_inst = np.where(goldB_cc[1]==i,1,0)
        
        inter_num = 0
        for j in range(1,pred_cc[0]+1):
            pred_inst = np.where(pred_cc[1]==j,1,0)
            union = goldB_inst + pred_inst
            if union.max() == 2:
                intersection = np.where(union==2,1,0)
                inter_num_curr = intersection.sum()
                if inter_num_curr > inter_num:
                    inter_num = inter_num_curr
                    union_chosen = union
                    
        if inter_num > 0:
            union_chosen = np.where(union_chosen>0,1,0)
            union_num = union_chosen.sum()
            if inter_num/union_num >= threshold:
                TP = TP + 1
                
    segmented = pred_cc[0] - 1
    positives = goldB_cc[0] - 1
    if segmented > 0:
        precision = TP / segmented
    else:
        precision = 0
    recall = TP / positives
    if precision+recall > 0:
        f_score = round(2*((precision*recall)/(precision+recall)),2)
    else:
        f_score = 0      
        
    return precision, recall, f_score


def getResultsIoU(first_out_channel, threshold, goldBinary_folder, ts_folder, model_name_pre, run_num):
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)
            
    image_names = os.listdir(ts_folder)
    cnt_f=0
    cnt_s=0
    results_sum = [0,0,0,0,0,0,0,0,0]
    
    if run_num==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(run_num) + '.pth'
                
    for name in image_names:
        tst_im_name = ts_folder + '/' + name
        gold_im_name = goldBinary_folder + '/' + name.split('.')[0] + '.png'
        im, _, prediction, goldB = hp.test(1, first_out_channel, model_name, tst_im_name, gold_im_name)

        precision, recall, f_score = IoU(prediction, goldB, threshold)

        results_sum[0] += precision
        results_sum[1] += recall
        results_sum[2] += f_score
        if fileTypesDict[name] == 'f':
            cnt_f+=1
            results_sum[3] += precision
            results_sum[4] += recall
            results_sum[5] += f_score
        else:
            cnt_s+=1
            results_sum[6] += precision
            results_sum[7] += recall
            results_sum[8] += f_score
    
    cnt = cnt_f+cnt_s
    results = [0,0,0,0,0,0,0,0,0]
    
    results[0] = round(results_sum[0]/cnt,4)
    results[1] = round(results_sum[1]/cnt,4)
    results[2] = round(results_sum[2]/cnt,4)
    
    results[3] = round(results_sum[3]/cnt_f,4)
    results[4] = round(results_sum[4]/cnt_f,4)
    results[5] = round(results_sum[5]/cnt_f,4)
    
    results[6] = round(results_sum[6]/cnt_s,4)
    results[7] = round(results_sum[7]/cnt_s,4)
    results[8] = round(results_sum[8]/cnt_s,4)   
    
    print('IoU')
    print(results)    
    return results

def IoU_accm(prediction, goldB, threshold):
    goldB_cc = cv2.connectedComponents(goldB)
    pred_cc = cv2.connectedComponents(prediction)
    TP = 0
    for i in range(1,goldB_cc[0]+1):
        goldB_inst = np.where(goldB_cc[1]==i,1,0)
        
        inter_num = 0
        for j in range(1,pred_cc[0]+1):
            pred_inst = np.where(pred_cc[1]==j,1,0)
            union = goldB_inst + pred_inst
            if union.max() == 2:
                intersection = np.where(union==2,1,0)
                inter_num_curr = intersection.sum()
                if inter_num_curr > inter_num:
                    inter_num = inter_num_curr
                    union_chosen = union
                    
        if inter_num > 0:
            union_chosen = np.where(union_chosen>0,1,0)
            union_num = union_chosen.sum()
            if inter_num/union_num >= threshold:
                TP = TP + 1
                
    segmented = pred_cc[0] - 1
    positives = goldB_cc[0] - 1     
        
    return segmented, positives, TP

def getPrecRecallFScore_IoU_accm(segmented,positives,TP):
    precision = TP/segmented
    recall = TP/positives
    f_score = 2*((precision*recall)/(precision+recall))
    return precision, recall, f_score

def getResultsIoU_accm(first_out_channel, threshold, goldBinary_folder, ts_folder, model_name_pre, run_num):
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)
            
    image_names = os.listdir(ts_folder)
    segmented_sum_f = 0
    positives_sum_f = 0
    TP_sum_f = 0
    segmented_sum_s = 0
    positives_sum_s = 0
    TP_sum_s = 0
    
    if run_num==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(run_num) + '.pth'
                
    for name in image_names:
        tst_im_name = ts_folder + '/' + name
        gold_im_name = goldBinary_folder + '/' + name.split('.')[0] + '.png'
        im, _, prediction, goldB = hp.test(1, first_out_channel, model_name, tst_im_name, gold_im_name)

        segmented, positives, TP = IoU_accm(prediction, goldB, threshold)

        if fileTypesDict[name] == 'f':
            segmented_sum_f = segmented_sum_f + segmented
            positives_sum_f = positives_sum_f + positives
            TP_sum_f = TP_sum_f + TP
        else:
            segmented_sum_s = segmented_sum_s + segmented
            positives_sum_s = positives_sum_s + positives
            TP_sum_s = TP_sum_s + TP
    
    segmented_sum = segmented_sum_f + segmented_sum_s
    positives_sum = positives_sum_f + positives_sum_s
    TP_sum = TP_sum_f + TP_sum_s
    results = [0,0,0,0,0,0,0,0,0]
    
    precision, recall, f_score = getPrecRecallFScore_IoU_accm(segmented_sum,positives_sum,TP_sum)    
    results[0] = round(precision,4)
    results[1] = round(recall,4)
    results[2] = round(f_score,4)
    
    precision, recall, f_score = getPrecRecallFScore_IoU_accm(segmented_sum_f,positives_sum_f,TP_sum_f)    
    results[3] = round(precision,4)
    results[4] = round(recall,4)
    results[5] = round(f_score,4)
    
    precision, recall, f_score = getPrecRecallFScore_IoU_accm(segmented_sum_s,positives_sum_s,TP_sum_s)    
    results[6] = round(precision,4)
    results[7] = round(recall,4)
    results[8] = round(f_score,4)  
    
    print('IoU_accumulated')
    print(results)    
    return results

def pixelwise(pred,gold):
    
    TP, FP, FN = pixelwiseFPFNTP(pred,gold)

    return getPrecRecallFScore(TP, FP, FN)

def getResultsPixelwise(first_out_channel, goldBinary_folder, ts_folder, model_name_pre, run_num):
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)
            
    image_names = os.listdir(ts_folder)
    cnt_f=0
    cnt_s=0
    results_sum = [0,0,0,0,0,0,0,0,0]
    
    if run_num==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(run_num) + '.pth'
                
    for name in image_names:
        tst_im_name = ts_folder + '/' + name
        gold_im_name = goldBinary_folder + '/' + name.split('.')[0] + '.png'
        im, _, prediction, goldB = hp.test(1, first_out_channel, model_name, tst_im_name, gold_im_name)

        [precision, recall, f_score] = pixelwise(prediction, goldB)

        results_sum[0] += precision
        results_sum[1] += recall
        results_sum[2] += f_score
        if fileTypesDict[name] == 'f':
            cnt_f+=1
            results_sum[3] += precision
            results_sum[4] += recall
            results_sum[5] += f_score
        else:
            cnt_s+=1
            results_sum[6] += precision
            results_sum[7] += recall
            results_sum[8] += f_score
            
    cnt = cnt_f+cnt_s
    results = [0,0,0,0,0,0,0,0,0]
    
    results[0] = round(results_sum[0]/cnt,4)
    results[1] = round(results_sum[1]/cnt,4)
    results[2] = round(results_sum[2]/cnt,4)
    
    results[3] = round(results_sum[3]/cnt_f,4)
    results[4] = round(results_sum[4]/cnt_f,4)
    results[5] = round(results_sum[5]/cnt_f,4)
    
    results[6] = round(results_sum[6]/cnt_s,4)
    results[7] = round(results_sum[7]/cnt_s,4)
    results[8] = round(results_sum[8]/cnt_s,4)   

    print('Pixelwise')    
    print(results)    
    return results
  
def get_weighted_hausdorff_dist(pred,gold):
    flag = False;
    if pred.sum() == 0:
        pred[0,:] = 1; pred[255,:] = 1; pred[:,0] = 1; pred[:,255] = 1;  
        flag = True
    gold_cc = cv2.connectedComponents(gold)
    pred_cc = cv2.connectedComponents(pred)
    return 0.5 * (wdh_one(pred_cc, gold_cc, pred,gold, flag) + wdh_one(gold_cc, pred_cc, gold,pred,flag))

def wdh_one(first_cc,second_cc,first,second, flag):
    whd = 0
    for i in range(1,first_cc[0]):
        first_inst = np.where(first_cc[1]==i,1,0)

        weight = first_inst.sum() / first.sum()

        if (first_inst * second).sum() > 0:
            intersection = 0
            for j in range(1,second_cc[0]):
                second_inst = np.where(second_cc[1]==j,1,0)
                intersection_new = (first_inst * second_inst).sum()
                if intersection_new > intersection:
                    hd = metrics.hausdorff_distance(first_inst==1, second_inst==1)
                    intersection = intersection_new
        else:
            if flag:
                hd = 0
                for j in range(1,second_cc[0]):
                    second_inst = np.where(second_cc[1]==j,1,0) 
                    hd_new = metrics.hausdorff_distance(first_inst==1, second_inst==1)
                    if hd_new > hd:
                        hd = hd_new                  
            else:
                hd = np.inf
                for j in range(1,second_cc[0]):
                    second_inst = np.where(second_cc[1]==j,1,0) 
                    hd_new = metrics.hausdorff_distance(first_inst==1, second_inst==1)
                    if hd_new < hd:
                        hd = hd_new  
    
        whd = whd + (weight * hd)   
    return whd        

def getResultsWeightdHausdorff(first_out_channel, goldBinary_folder, ts_folder, model_name_pre, run_num):
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)
            
    image_names = os.listdir(ts_folder)
    cnt_f=0
    cnt_s=0
    results_sum = [0,0,0]
    
    if run_num==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(run_num) + '.pth'
                
    for name in image_names:
        tst_im_name = ts_folder + '/' + name
        gold_im_name = goldBinary_folder + '/' + name.split('.')[0] + '.png'
        im, _, prediction, goldB = hp.test(1, first_out_channel, model_name, tst_im_name, gold_im_name)

        whd = get_weighted_hausdorff_dist(prediction,goldB)

        results_sum[0] += whd
        if fileTypesDict[name] == 'f':
            cnt_f+=1
            results_sum[1] += whd
        else:
            cnt_s+=1
            results_sum[2] += whd
            
    cnt = cnt_f+cnt_s
    results = [0,0,0]
    
    results[0] = round(results_sum[0]/cnt,4)
    results[1] = round(results_sum[1]/cnt_f,4)
    results[2] = round(results_sum[2]/cnt_s,4)   

    print('Hausdorff')    
    print(results)    
    return results

def create_result_excel(num_of_runs, threshold, goldBinary_folder, ts_folder, model_name_pre, excel_name):
    import xlsxwriter
    
    workbook = xlsxwriter.Workbook(excel_name)
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, "IoU");
    worksheet.write(0, 1, "avrg_precision"); worksheet.write(0, 2, "avrg_recall"); 
    worksheet.write(0, 3, "avrg_f_score");
    worksheet.write(0, 4, "avrg_precision_f"); worksheet.write(0, 5, "avrg_recall_f"); 
    worksheet.write(0, 6, "avrg_f_score_f");  
    worksheet.write(0, 7, "avrg_precision_s"); worksheet.write(0, 8, "avrg_recall_s"); 
    worksheet.write(0, 9, "avrg_f_score_s");  
    
    for i in range(1,num_of_runs+1):        
        resultIoU = getResultsIoU(32, threshold, goldBinary_folder, ts_folder, model_name_pre, i)
        
        worksheet.write(i, 0, "run_"+str(i)); 
        
        for j in range(0,9):
            worksheet.write(i, j+1, resultIoU[j]);

    ii=i+1        
    worksheet.write(ii, 0, "IoU_accm");
    for i in range(1,num_of_runs+1):        
        resultIoU = getResultsIoU_accm(32, threshold, goldBinary_folder, ts_folder, model_name_pre, i)
        
        worksheet.write(ii+i, 0, "run_"+str(i)); 
        
        for j in range(0,9):
            worksheet.write(ii+i, j+1, resultIoU[j]);            
    
    iii=ii+i+1
    worksheet.write(iii, 0, "Pixelwise");  
    
    for i in range(1,num_of_runs+1):        
        resultPixelwise = getResultsPixelwise(32, goldBinary_folder, ts_folder, model_name_pre, i)
        
        worksheet.write(iii+i, 0, "run_"+str(i)); 
        
        for j in range(0,9):
            worksheet.write(iii+i, j+1, resultPixelwise[j]);    
    
    iiii=iii+i+1
    worksheet.write(iiii, 0, "WeightdHausdrff");
    worksheet.write(iiii, 1, "all"); worksheet.write(iiii, 2, "big"); worksheet.write(iiii, 3, "small");
    
    for i in range(1,num_of_runs+1):        
        resultWhd = getResultsWeightdHausdorff(32, goldBinary_folder, ts_folder, model_name_pre, i)
        
        worksheet.write(iiii+i, 0, "run_"+str(i)); 
        
        for j in range(0,3):
            worksheet.write(iiii+i, j+1, resultWhd[j]);      
    
    workbook.close()


