import cg_tm_kl
log_file = r"D:\Destop\seqs\660kb_3300\read_in_4\{}.txt".format('[29]-3')
lines_ = cg_tm_kl.txt_process_sc_duo(log_file,len_sc=200,beg_sc=0,end_sc=3300,PADDING=False,flex=0,num1=200)

with open(r'D:\Destop\seqs\660kb_3300\read_in_4\pp\[29]-3.txt','w') as f1:
    for line in  lines_:
        f1.write(line)
        f1.write('\n')
