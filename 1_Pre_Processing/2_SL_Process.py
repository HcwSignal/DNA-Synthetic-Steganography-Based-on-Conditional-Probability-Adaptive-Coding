import cg_tm_kl
import os

def main(Path):

    if Path.find('198') > 1:
        index = ['3','6','33','66']
        len_sc_ = 198
    else:
        index = ['2', '4', '5', '8', '10', '20', '40', '50']
        len_sc_ = 200

    for len_seq_ in index:
        lines = cg_tm_kl.txt_process_sc_duo(dp_sc=Path, len_sc=len_sc_, beg_sc=0, end_sc=9999999, PADDING=False, flex=0,
                                            num1=int(len_seq_))

        P_write = Path[: -4] + '_' + str(len_seq_) + '.txt'
        with open(P_write, 'w') as f1:
            for line in lines:
                f1.write(line)
                f1.write('\n')

if __name__ == '__main__':
    name = [18,28,43]

    for n in name:
        FileAll = r'D:\Destop\file\additional data\{}\base_extaction'.format(str(n))
        Path = []
        for root, dirs, files in os.walk(FileAll):
            for file in files:
                Path.append(os.path.join(root, file))

        for p in Path:
            main(p)
