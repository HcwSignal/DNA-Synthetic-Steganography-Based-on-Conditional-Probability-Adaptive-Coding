import rf_functinn
import supprting_function
import pandas as pd
import os
for i in ['611']:
    PATH_generated_path = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\baseline_xiaorong'.format(i)

    PATH_generated = []
    for root, dirs, files in os.walk(PATH_generated_path):
        for file in files:
            PATH_generated.append(os.path.join(root, file))

    All_RF = pd.DataFrame(columns=['path', 'accurcy'])

    for p in PATH_generated:
        file_g = p
        file_original = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\OriginalData\{}.txt'.format(i, i)
        if i == '660' or i == '888':
            SeqLength = 198
            ind = '3'
        else:
            SeqLength = 200
            ind = '4'
        for j in range(1):
            t = p.rfind('\\')
            name = p[t + 1: len(p) - 4]
            accurcy = rf_functinn.rf(file_g, file_original, lensc=SeqLength, ind=ind)
            All_RF = All_RF.append([{'path': name, 'accurcy': accurcy}], ignore_index=True)

        print(1)

        accurcy_average, accurcy_up, accurcy_down, accurcy_std = supprting_function.data_analysis(All_RF,
                                                                                                  type='accurcy')

        All_RF = All_RF.append([{'path': 'ave', 'accurcy': accurcy_average}], ignore_index=True)
        All_RF = All_RF.append([{'path': 'ACC_STD', 'accurcy': accurcy_std}], ignore_index=True)
        All_RF = All_RF.append([{'path': 'Pos ERR', 'accurcy': accurcy_up}], ignore_index=True)
        All_RF = All_RF.append([{'path': 'Neg ERR', 'accurcy': accurcy_down}], ignore_index=True)

        print(All_RF)

    All_RF.to_csv(r'D:\Desktop\PaperCode_2023\ExperimentData\{}}\{}_RF.csv'.format(i, i))



