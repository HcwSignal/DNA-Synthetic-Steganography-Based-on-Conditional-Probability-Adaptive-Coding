import random

import rf_function
import os
import pathlib
import pandas as pd
import logger
from logger import Logger
import cg_tm_kl
import numpy as np
import supprting_function

if __name__ == '__main__':
	FILE_mode = 'a' #"b":baselin mode; "a":ADG mode

	Classifier_modes = {}

	Classifier_modes['RF'] = True

	Classifier_modes['RNNs'] = False

	#Classifier_modes['LSTM'] = True


	mode_file_name = {}

	mode_file_name['160'] = r'ena_178901_noncoding'
	mode_file_name['221'] = r'ena_1909293_noncoding'
	mode_file_name['572'] = r'ena_438_noncoding'
	mode_file_name['611'] = r'ena_39491_noncoding'

	#modes = ['160','221','572']  #synthesis_sequence typy index
	modes = ['611']
	for mode in modes:
		Path_result_save = r'D:\Desktop\PaperCode_2023\ExperimentData\{}'.format(mode)

		# RF分类结果存放路径
		pathlib.Path(Path_result_save + r'\RF').mkdir(parents=True, exist_ok=True)  # 父目录不存在则创建父目录；若目录存在，不产生error
		# Baseline方法存放路径
		pathlib.Path(Path_result_save + r'\baselines').mkdir(parents=True, exist_ok=True)  # 父目录不存在则创建父目录；若目录存在，不产生error
		# csv文件存放路径
		file_csv = Path_result_save + r'\csv'
		pathlib.Path(file_csv).mkdir(parents=True, exist_ok=True)
		# RNNS分类结果存放路径
		# pathlib.Path(Path_result_save + r'\S-I\RNNs').mkdir(parents=True, exist_ok=True)  # 父目录不存在则创建父目录；若目录存在，不产生error
		# RNNs_Result_csv_file = Path_result_save + r'\S-I\RNNs\{}.csv'.format(mode)

		if FILE_mode == 'a':
			index = ['2','3','4','5','6','8','10','20','33','40','50','66']
			#index = ['4']
			file_finaly = Path_result_save + r'\RF\{}_ADG_.csv'.format(mode)  # 所有文件的得分值集合

		else:
			index = ['4']  # 在优选的SL下生成baselines，二分类任务时只需要固定一个index即可

			file_finaly = Path_result_save + r'\RF\{}_baselines_.csv'.format(mode)  # 所有文件的得分值集合

		All_RF = pd.DataFrame(columns=['path', 'accurcy'])

		# All_RNNs = pd.DataFrame(columns=['path', 'accurcy', 'reacll_score', 'pres', 'f1s', 'bpn', 'ebpn'])

		for ind in index:
			temp = pd.DataFrame(columns=['path', 'accurcy'])
			if FILE_mode == 'a':
				log_file = Path_result_save + r'\RF_ADG_log.txt'  # 过程记录

				file_generated = Path_result_save + r'\seq\{}'.format(ind)  # 生成文件的存放文件夹位置
			else:
				log_file = Path_result_save + r"\RF\RF_baselines_log.txt"

				file_generated = Path_result_save + r'\baselines'

			logger = Logger(log_file)

			PATH_generated = []

			BPN = {}

			if int(ind) % 3 == 0:
				SeqLength = 198
				file_original = Path_result_save + r'\base_extaction\{}_198_{}.txt'.format(str(mode_file_name[mode]),
																						   ind)
			else:
				SeqLength = 200
				file_original = Path_result_save + r'\base_extaction\{}_200_{}.txt'.format(str(mode_file_name[mode]),
																						   ind)

			with open(file_original, 'r') as f1:
				lines = f1.readlines()

			for root, dirs, files in os.walk(file_generated):
				for file in files:
					PATH_generated.append(os.path.join(root, file))

			if Classifier_modes['RF'] == True:
				for file_g in PATH_generated:
					p_ = file_g[file_g.rfind('\\') + 1:  len(file_g) - 4]
					supprting_function.csv_shengc(file_g, file_csv, file_original, len(lines), Seqlength=SeqLength)
					BPN[p_] = cg_tm_kl.find_bpn(file_g)
					accurcy = rf_function.rf(file_g, file_original, endsc=len(lines), lensc=SeqLength,ind =ind)
					#accurcy = random.randint(0,100) #test
					temp = temp.append([{'path': p_, 'accurcy': accurcy}], ignore_index=True)
					All_RF = All_RF.append([{'path': p_, 'accurcy': accurcy}], ignore_index=True)

				accurcy_average, accurcy_up, accurcy_down, accurcy_std = supprting_function.data_analysis(temp,
																										  type='accurcy')

				All_RF = All_RF.append([{'path': 'ave', 'accurcy': accurcy_average}], ignore_index=True)
				All_RF = All_RF.append([{'path': 'ACC_STD', 'accurcy': accurcy_std}], ignore_index=True)
				All_RF = All_RF.append([{'path': 'Pos ERR', 'accurcy': accurcy_up}], ignore_index=True)
				All_RF = All_RF.append([{'path': 'Neg ERR', 'accurcy': accurcy_down}], ignore_index=True)

			'''
			if Classifier_modes['RNNs'] == True:
				for file_g in file_csv:
					p_ = file_g[file_g.rfind('\\') + 1:  len(file_g) - 4]
					rnns_accurcy, rnns_reaclls, rnns_pres, rnns_f1s = RNNs.RNNs_Classifier(file_g)
					# rnns_accurcy, rnns_reaclls, rnns_pres, rnns_f1s = 0,0,0,0 #test
					bpn = float(BPN[p_])
					ebpn = (1 - rnns_accurcy) * bpn * 2
					temp = temp.append([{'path': p_, 'accurcy': rnns_accurcy, 'reacll_score': rnns_reaclls,
												 'pres': rnns_pres, 'f1s': rnns_f1s,
												 'bpn': bpn, 'ebpn': ebpn}], ignore_index=True)
					All_RNNs = All_RNNs.append([{'path': p_, 'accurcy': rnns_accurcy, 'reacll_score': rnns_reaclls,
												 'pres': rnns_pres, 'f1s': rnns_f1s,
												 'bpn': bpn, 'ebpn': ebpn}], ignore_index=True)

				rnns_accurcy_average, rnns_accurcy_up, rnns_accurcy_down, rnns_accurcy_std= supprting_function.data_analysis(temp ,type='accurcy')
				rnns_reacll_score_average, rnns_reacll_score_up, rnns_reacll_score_down, rnns_reacll_score_std = supprting_function.data_analysis(
					temp, type='reacll_score')
				rnns_pres_average, rnns_pres_up, rnns_pres_down, rnns_pres_std= supprting_function.data_analysis(temp, type='pres')
				rnns_f1s_average, rnns_f1s_up, rnns_f1s_down, rnns_f1s_std= supprting_function.data_analysis(temp
																											  , type='f1s')

				All_RNNs = All_RNNs.append([{'path': 'ave', 'accurcy': rnns_accurcy_average
												,'recall_scoe' :rnns_reacll_score_average,'pres' :rnns_pres_average,'f1s' :rnns_f1s_average}], ignore_index=True)
				All_RNNs = All_RNNs.append([{'path': 'ave', 'accurcy': rnns_accurcy_std
												,'recall_score':rnns_reacll_score_std,'pres':rnns_pres_std,'f1s':rnns_f1s_std}], ignore_index=True)
			'''
		if Classifier_modes['RF'] == True:
			All_RF.to_csv(file_finaly)
		# if Classifier_modes['RNNs'] == True:
		# 	All_RNNs.to_csv(RNNs_Result_csv_file)

