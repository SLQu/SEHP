


import torch


"""
| Model  | AUC | SNS |  MNS | CNS | MIXED |  AVG | AP | SNS |  MNS | CNS | MIXED |  AVG | Score | Pos | SNS |  MNS | CNS | MIXED |  AVG  | Diff | SNS |  MNS | CNS | MIXED |  AVG  |
| ---    | --- | --- | ---  | ---  | ---  | ---  | ---| --- | ---  |  ---| ---   | ---  |  ---  | --- | --- | ---  | --- |  ---  | ---   | ---  |---  | ---  | --- | ---   | ---   | 
|ED |    | 0.5204 | 0.5850 | 0.4515 | 0.5244 | 0.5203 |    | 0.5000 | 0.8333 | 0.5000 | 0.5556 | 0.5972 |    | 0.3831    | 0.3791 | 0.3672 | 0.3906 | 0.3776 | 0.3786 |    | 0.0040 | 0.0160 | -0.0075 | 0.0055 | 0.0045 |  
"""


"""
| Model  | AUC | AP | Pos|  Neg  | Diff |
| ---    | --- | ---| --- | --- | ---  | 
|ED    | 0.5204 | 0.5850 | 0.4515 | 0.5244 | 0.5203 | 
"""


def out(result):
    print('-------------------')
    print('\n\n\n')

    _out_ = f"{result['data']} \n|{result['model']} |   "
    auc, ap , n_score, p_score , diff = [], [], [], [], []


    candidate = []
    for n_key in ['sns', 'mns', 'cns', 'mixed']:
        if n_key in result:
            candidate.append(n_key)

    for n_key in candidate:
        auc.append(result[f'{n_key}']['auroc'])
        ap.append(result[f'{n_key}']['precision'])
        if type(result[f'{n_key}']['neg']) is torch.Tensor:
            result[f'{n_key}']['neg'] = result[f'{n_key}']['neg'].item()
            result[f'{n_key}']['pos'] = result[f'{n_key}']['pos'].item()
            result[f'{n_key}']['diff'] = result[f'{n_key}']['diff'].item()
            

        n_score.append(result[f'{n_key}']['neg'])
        p_score.append(result[f'{n_key}']['pos'])
        diff.append(result[f'{n_key}']['diff']) 


    if len(candidate) == 4:
        for i in auc:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(auc)/4:.4f} |   "

        for i in ap:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(ap)/4:.4f} |   "

        _out_ += f" | {p_score[0]:.4f}   "

        for i in n_score:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(n_score)/4:.4f} |   "


        for i in diff:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(diff)/4:.4f} |   "

    else:
        _out_ += f" {auc[0]:.4f} "
        _out_ += f" | {ap[0]:.4f} "
        _out_ += f" | {p_score[0]:.4f} "
        _out_ += f" | {n_score[0]:.4f} "
        _out_ += f" | {diff[0]:.4f} |   "
    print(_out_)

    print('\n\n\n')
    print('-------------------')
    return _out_
    


def out_all(result):
    print('-------------------')
    print('accuracy, f1, auroc, precision, recall')
    print('\n\n\n')

    _out_ = f"{result['data']} \n|{result['model']} |   "
    auc, ap , n_score, p_score , diff = [], [], [], [], []
    accuracy, auroc, f1, precision, recall = [], [], [], [], []


    candidate = []
    for n_key in ['sns', 'mns', 'cns', 'mixed']:
        if n_key in result:
            candidate.append(n_key)

    for n_key in candidate:
        accuracy.append(result[f'{n_key}']['accuracy'])
        f1.append(result[f'{n_key}']['f1'])
        auroc.append(result[f'{n_key}']['auroc'])
        precision.append(result[f'{n_key}']['precision'])
        recall.append(result[f'{n_key}']['recall'])
        if type(result[f'{n_key}']['neg']) is torch.Tensor:
            result[f'{n_key}']['neg'] = result[f'{n_key}']['neg'].item()
            result[f'{n_key}']['pos'] = result[f'{n_key}']['pos'].item()
            result[f'{n_key}']['diff'] = result[f'{n_key}']['diff'].item()
            

        n_score.append(result[f'{n_key}']['neg'])
        p_score.append(result[f'{n_key}']['pos'])
        diff.append(result[f'{n_key}']['diff']) 


    if len(candidate) == 4:
        for i in auc:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(auc)/4:.4f} |   "

        for i in ap:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(ap)/4:.4f} |   "

        _out_ += f" | {p_score[0]:.4f}   "

        for i in n_score:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(n_score)/4:.4f} |   "


        for i in diff:
            _out_ += f" | {i:.4f}"
        _out_ += f" | {sum(diff)/4:.4f} |   "

    else:
        _out_ += f" {accuracy[0]:.4f} "
        _out_ += f" | {f1[0]:.4f} "
        _out_ += f" | {auroc[0]:.4f} "
        _out_ += f" | {precision[0]:.4f} "
        _out_ += f" | {recall[0]:.4f} "
        _out_ += f" |              {p_score[0]:.4f} "
        _out_ += f" | {n_score[0]:.4f} "
        _out_ += f" | {diff[0]:.4f} |   "
    print(_out_)

    print('\n\n\n')
    print('-------------------')
    return _out_
    

# n_score = result[f'{n_key}']['neg'].item()
# p_score = result[f'{n_key}']['pos'].item()
# auc = result[f'{n_key}']['auroc']
# ap = result[f'{n_key}']['precision']
# print(f"{n_key},(auc, ap):({auc:.4f}, {ap:.4f}), (pos,neg,diff): ({p_score:.4f},{n_score:.4f},{(p_score - n_score):.4f}) ")