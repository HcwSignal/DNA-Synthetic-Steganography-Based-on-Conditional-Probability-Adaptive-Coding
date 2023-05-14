import cg_tm_kl
import os

def main(FileAll):
    lens_seq = [198,200]
    Path = []
    for root, dirs, files in os.walk(FileAll):
        for file in files:
            Path.append(os.path.join(root, file))

    for len_seq_ in lens_seq:
        for P in Path:
            lines = cg_tm_kl.BaseExtraaction(P, len_single=len_seq_, batchsize=1)

            P_write = P[: -4] + '_' + str(len_seq_) + '.txt'
            with open(P_write, 'w') as f1:
                for line in lines:
                    f1.write(line)
                    f1.write('\n')


if __name__ == '__main__':
    main(r'D:\Destop\file\科研相关\论文代码\ExperimentData\572\Original')