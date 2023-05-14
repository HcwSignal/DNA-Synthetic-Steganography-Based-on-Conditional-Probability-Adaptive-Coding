import os
def qingchu_Model(path):
    File = []

    for root, dirs, files in os.walk(path):
        for file in files:
            File.append(os.path.join(root, file))

    Models = []
    for f in File:
        if f.find('pkl') > 0:
            Models.append(f)
    m_name = []
    epoch_nums = []
    for m in Models:
        m_ = m[m.rfind('\\') + 1 : -4]
        m_name = m_.split('-')
        epoch_nums.append(int(m_name[-1]))
        #for elements in m_name:
        #    try:
        #        epoch_nums.append(int(elements))
        #    except:
        #        pattern = elements

    save_modle = m_name[0] + '-' + m_name[1] + '-' + str(max(epoch_nums))
    for f in Models:
        if f.find(save_modle) > 0:
            continue

        os.remove(f)